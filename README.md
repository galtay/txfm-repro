# txfm-repro

Minimal reproduction of **TxFM** (Kenyon-Dean et al., ICLR 2026 Workshop on
Foundation Models for Science) — a transformer masked-autoencoder for gene
expression. Paper PDF lives at `reference/36_Effective_Biological_Repres.pdf`;
short orientation in `reference/notes.md`.

## Phase 0 — model scaffold on mock data

This phase stands up a clean Lightning-driven scaffold. Real bulk RNA-seq
(via the `tcga2hf` HuggingFace dataset) gets wired in next.

```bash
uv sync
uv run pytest -q
uv run txfm-cli fit --config configs/mock.yaml
```

Override anything from the YAML on the CLI:

```bash
uv run txfm-cli fit --config configs/mock.yaml --trainer.max_epochs 2 --data.batch_size 8
```

## Layout

- `src/txfm_repro/lit_model.py` — `LitTxFMConfig`, the architectural
  components (encoder, decoder, rectified-tanh activation), the Poisson loss,
  and the `LitTxFM` `LightningModule`.
- `src/txfm_repro/mock_data.py` — synthetic bulk RNA-seq counts +
  `MockBulkDataModule`.
- `src/txfm_repro/cli.py` — `LightningCLI` entrypoint with `link_arguments`
  syncing shared knobs between data and model.
- `configs/mock.yaml` — default training config.
- `tests/test_model.py` — shape/bound checks + overfit-a-batch smoke test.
