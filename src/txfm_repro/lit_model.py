"""TxFM model components, loss, and LightningModule.

Faithful but minimal port of the architecture in
`reference/36_Effective_Biological_Repres.pdf` (Kenyon-Dean et al., ICLR 2026
Workshop FM4S):

  - encoder: transformer over a small subset of unmasked genes, each
    represented as `[gene_embedding (d-1) ; preprocessed_value (1)]`,
    plus a learnable CLS token whose final embedding is the sample
    representation.
  - decoder: small MLP on the CLS embedding, predicting all G genes.
  - output activation: rectified tanh, paper Eq. 1.
  - loss: per-gene Poisson, paper Eq. 2.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import lightning as L
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ActivationName = Literal["tanh", "sigmoid"]
LossReduction = Literal["none", "mean", "sum"]


@dataclass
class LitTxFMConfig:
    """Model-side hyperparameters.

    `n_genes` and `library_size_L` are shared with the data module; the
    Lightning CLI links them so they only need to be set once in the
    YAML's `data:` block.
    """

    n_genes: int = 2000
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dim_ff: int = 512
    decoder_layers: int = 2
    dropout: float = 0.1
    library_size_L: float = 1e5
    # Output activation form. The two forms are mathematically equivalent
    # (paper Eq. 1's rectified_tanh = the opentxfm reference repo's
    # RectifiedSigmoid via tanh(y) = 2*sigmoid(2y) - 1) but may differ in
    # numerical / runtime characteristics. Default = "tanh" to match the
    # paper's notation.
    activation: ActivationName = "tanh"
    # Loss reduction. "mean" matches our original default; "none" leaves
    # the per-element tensor for caller-side weighting (matches the
    # opentxfm reference repo's PoissonNLLLogSpace style).
    loss_reduction: LossReduction = "mean"


def rectified_tanh(z: Tensor, library_size_L: float) -> Tensor:
    """Paper Eq. 1: log(L+1) * ReLU(tanh(z / (4e))).

    Bounded in [0, log(L+1)], smoothly saturating; respects the count nature
    of the data without explicit zero inflation.
    """
    return math.log1p(library_size_L) * F.relu(torch.tanh(z / (4.0 * math.e)))


def rectified_sigmoid(z: Tensor, library_size_L: float) -> Tensor:
    """Equivalent surface form of Eq. 1: log(L+1) * ReLU(2*sigmoid(z/(2e)) - 1).

    Identity used: tanh(y) = 2*sigmoid(2y) - 1, with y = z / (4e), giving
    tanh(z/(4e)) = 2*sigmoid(z/(2e)) - 1. Matches the opentxfm reference
    repo's `RectifiedSigmoid`. Output is identical to `rectified_tanh` up to
    floating-point rounding; `tests/test_model.py::test_activation_equivalence`
    asserts that. Kept side-by-side so we can A/B test runtime + numerics.
    """
    return math.log1p(library_size_L) * F.relu(2.0 * torch.sigmoid(z / (2.0 * math.e)) - 1.0)


def get_output_activation(name: ActivationName):
    if name == "tanh":
        return rectified_tanh
    if name == "sigmoid":
        return rectified_sigmoid
    raise ValueError(f"unknown activation {name!r}; expected 'tanh' or 'sigmoid'")


class TxFMEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model < 2:
            raise ValueError("d_model must be >= 2 to leave room for the value scalar")
        self.gene_emb = nn.Embedding(n_genes, d_model - 1)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        unmasked_idx: Tensor,
        unmasked_vals: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        # unmasked_idx, unmasked_vals: (B, K) — K may be larger than per-row
        # actual sequence length when this batch mixes samples with different
        # gene-coverage / mask sizes; pad slots are flagged in padding_mask.
        # padding_mask: (B, K) bool, True at pad slots; pad slots are excluded
        # from attention. Index/value content at pad slots is don't-care.
        token_emb = self.gene_emb(unmasked_idx)                    # (B, K, d-1)
        tokens = torch.cat([token_emb, unmasked_vals.unsqueeze(-1)], dim=-1)  # (B, K, d)
        cls = self.cls.expand(tokens.size(0), -1, -1)               # (B, 1, d)
        x = torch.cat([cls, tokens], dim=1)                         # (B, K+1, d)
        if padding_mask is not None:
            # CLS slot is always live, so prepend a False column to the mask.
            cls_mask = torch.zeros(
                padding_mask.size(0), 1, dtype=torch.bool, device=padding_mask.device,
            )
            src_key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            src_key_padding_mask = None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x[:, 0, :]                                           # (B, d)


class MLPDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_genes: int,
        decoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be >= 1")
        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(decoder_layers - 1)
            ]
        )
        self.out = nn.Linear(d_model, n_genes)

    def forward(self, e: Tensor) -> Tensor:
        x = e
        for block in self.hidden:
            x = x + block(x)  # residual
        return self.out(x)


class TxFM(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_ff: int,
        decoder_layers: int,
        dropout: float,
        library_size_L: float,
        activation: ActivationName = "tanh",
    ) -> None:
        super().__init__()
        self.library_size_L = library_size_L
        self.activation_name = activation
        self._activation = get_output_activation(activation)
        self.encoder = TxFMEncoder(
            n_genes=n_genes,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.decoder = MLPDecoder(
            d_model=d_model,
            n_genes=n_genes,
            decoder_layers=decoder_layers,
            dropout=dropout,
        )

    def forward(
        self,
        unmasked_idx: Tensor,
        unmasked_vals: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        e = self.encoder(unmasked_idx, unmasked_vals, padding_mask=padding_mask)
        z = self.decoder(e)
        return self._activation(z, self.library_size_L)


def poisson_loss(
    x_hat_log: Tensor,
    x_target_log: Tensor,
    reduction: LossReduction = "mean",
    target_mask: Tensor | None = None,
) -> Tensor:
    """Paper Eq. 2: per-gene Poisson loss in the log-rate parameterization.

    Per-element form: `e^x_hat - x_hat * e^x_target`. Both inputs live in
    the library-normalized log1p space. `reduction` follows torch convention:
    `"none"` returns the per-element tensor (matches the opentxfm reference
    repo's `PoissonNLLLogSpace(reduction="none")`); `"mean"` and `"sum"`
    reduce over all axes.

    If `target_mask` is provided (bool, same shape as inputs), the loss only
    accumulates over positions where the mask is True — used when samples
    come from sources with different measured gene sets, so unmeasured
    positions have no ground truth and should not contribute. With
    `reduction="mean"` the denominator is `target_mask.sum()`, not numel.
    With `reduction="none"` the masked positions are zeroed out (callers can
    still reduce however they like).
    """
    per_elem = x_hat_log.exp() - x_hat_log * x_target_log.exp()
    if target_mask is not None:
        m = target_mask.to(per_elem.dtype)
        per_elem = per_elem * m
        if reduction == "mean":
            denom = m.sum().clamp_min(1.0)
            return per_elem.sum() / denom
        if reduction == "sum":
            return per_elem.sum()
        if reduction == "none":
            return per_elem
        raise ValueError(f"unknown reduction {reduction!r}; expected 'none', 'mean', or 'sum'")
    if reduction == "none":
        return per_elem
    if reduction == "mean":
        return per_elem.mean()
    if reduction == "sum":
        return per_elem.sum()
    raise ValueError(f"unknown reduction {reduction!r}; expected 'none', 'mean', or 'sum'")


class LitTxFM(L.LightningModule):
    """Lightning wrapper. Optimizer is configured via the Lightning CLI's
    `optimizer:` YAML block — no `configure_optimizers` defined here on
    purpose, so the YAML must declare one explicitly during scaffolding."""

    def __init__(self, cfg: LitTxFMConfig) -> None:
        super().__init__()
        # Lightning round-trips hparams as dicts on `load_from_checkpoint`,
        # so `cfg` may arrive here as a `dict` instead of the dataclass.
        # The annotation stays `LitTxFMConfig` so jsonargparse-driven CLI
        # linking (`data.init_args.* -> model.cfg.*`) still works; we just
        # normalize at runtime.
        if isinstance(cfg, dict):
            cfg = LitTxFMConfig(**cfg)
        self.save_hyperparameters({"cfg": asdict(cfg)})
        self.cfg = cfg
        self.model = TxFM(
            n_genes=cfg.n_genes,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dim_ff=cfg.dim_ff,
            decoder_layers=cfg.decoder_layers,
            dropout=cfg.dropout,
            library_size_L=cfg.library_size_L,
            activation=cfg.activation,
        )

    def forward(
        self,
        unmasked_idx: Tensor,
        unmasked_vals: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        return self.model(unmasked_idx, unmasked_vals, padding_mask=padding_mask)

    def _step(self, batch: dict[str, Tensor], prefix: str) -> Tensor:
        x_hat = self(
            batch["unmasked_idx"],
            batch["unmasked_vals"],
            padding_mask=batch.get("padding_mask"),
        )
        target_mask = batch.get("target_mask")
        # `mean` here = mean over measured positions when target_mask is
        # provided, otherwise mean over all elements. Used both for logging
        # (always) and as the backward objective when cfg.loss_reduction == "mean".
        loss_mean = poisson_loss(
            x_hat, batch["target"], reduction="mean", target_mask=target_mask,
        )
        self.log(f"{prefix}/loss", loss_mean, prog_bar=True, on_epoch=True, on_step=(prefix == "train"))

        # Validation-only richer metrics: splits loss into visible/holdout,
        # plus per-sample Pearson + R² on held-out positions. Skipped during
        # training to keep the step cheap.
        if prefix == "val":
            from txfm_repro.metrics import compute_holdout_metrics
            extra = compute_holdout_metrics(
                x_hat=x_hat,
                target=batch["target"],
                unmasked_idx=batch["unmasked_idx"],
                padding_mask=batch.get("padding_mask"),
                target_mask=target_mask,
            )
            for k, v in extra.items():
                if torch.isnan(v):
                    continue
                self.log(f"val/{k}", v, on_epoch=True, on_step=False)

        if self.cfg.loss_reduction == "mean":
            return loss_mean
        if self.cfg.loss_reduction == "sum":
            return poisson_loss(
                x_hat, batch["target"], reduction="sum", target_mask=target_mask,
            )
        # "none" — caller asked for unreduced, but Lightning needs a scalar
        # for backward; fall back to mean.
        return loss_mean

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def predict_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor | list[str]]:
        """Return the CLS embedding `s` (encoder output) — no decoder, no loss.

        The batch dict may carry an optional `case_id` list (a `metadata_collate`
        path attaches it for the predict dataloader); when present it's
        passed through so the caller can pair embeddings with patient IDs.
        """
        s = self.model.encoder(
            batch["unmasked_idx"],
            batch["unmasked_vals"],
            padding_mask=batch.get("padding_mask"),
        )
        out: dict[str, Tensor | list[str]] = {"embedding": s}
        if "case_id" in batch:
            out["case_id"] = batch["case_id"]
        return out
