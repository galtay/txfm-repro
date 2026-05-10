"""Lightning CLI entry point.

Run `txfm-cli fit --config configs/mock.yaml` after `uv sync`.
"""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from txfm_repro.lit_model import LitTxFM
from txfm_repro.mock_data import MockBulkDataModule


class TxFMCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # Single source of truth for shared knobs — set them in `data:`
        # and they propagate to `model.cfg`.
        parser.link_arguments("data.n_genes", "model.cfg.n_genes")
        parser.link_arguments("data.library_size_L", "model.cfg.library_size_L")


def main() -> None:
    TxFMCLI(
        model_class=LitTxFM,
        datamodule_class=MockBulkDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
