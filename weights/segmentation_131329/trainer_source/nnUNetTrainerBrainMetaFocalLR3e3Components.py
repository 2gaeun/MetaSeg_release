"""Focal BrainMetaSeg trainer with a lower 3e-3 initial learning rate and component logging."""

import torch

from brainmetaseg.nnunet_ext.nnUNetTrainerBrainMetaFocal import (
    nnUNetTrainerBrainMetaFocal,
)


class nnUNetTrainerBrainMetaFocalLR3e3Components(nnUNetTrainerBrainMetaFocal):
    """Log each loss term while keeping the focal LR experiment fixed."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-3

    def on_train_start(self):
        result = super().on_train_start()
        if self.local_rank == 0:
            self.logger.log_summary("training/initial_lr", float(self.initial_lr))
            self.print_to_log_file(
                f"Initial learning rate override: {self.initial_lr}",
                also_print_to_console=True,
            )
        return result
