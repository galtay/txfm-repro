"""Lightning CLI entry point.

The datamodule is registered in *subclass mode* so any LightningDataModule
subclass can be picked from YAML via `data.class_path` + `data.init_args`.
That keeps a single entry point (`txfm-cli`) usable with both the mock
data module and the TCGA streaming one.

Run examples:
  txfm-cli fit --config configs/mock.yaml
  txfm-cli fit --config configs/tcga_luad.yaml
"""

from __future__ import annotations

import lightning as L
from lightning.pytorch.cli import LightningCLI

from txfm_repro.lit_model import LitTxFM


class TxFMCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # Single source of truth for shared knobs — set them in
        # `data.init_args` and they propagate to `model.cfg`.
        parser.link_arguments("data.init_args.n_genes", "model.cfg.n_genes")
        parser.link_arguments("data.init_args.library_size_L", "model.cfg.library_size_L")


def main() -> None:
    TxFMCLI(
        model_class=LitTxFM,
        datamodule_class=L.LightningDataModule,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
