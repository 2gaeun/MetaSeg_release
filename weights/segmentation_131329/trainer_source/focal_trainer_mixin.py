"""Shared focal-loss policy for BrainMetaSeg nnU-Net trainer variants."""

from __future__ import annotations

from brainmetaseg.nnunet_ext.focal_loss import (
    BrainMetaFocalLoss,
    add_focal_to_nnunet_loss,
)


class BrainMetaFocalTrainerMixin:
    """Add the paper-configured focal term without rebuilding nnU-Net's loss."""

    focal_alpha = 0.25
    focal_gamma = 2.0
    focal_weight = 0.5

    def _build_loss(self):
        base_loss = super()._build_loss()
        focal_loss = BrainMetaFocalLoss(
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            has_regions=self.label_manager.has_regions,
            ignore_label=self.label_manager.ignore_label,
        )
        return add_focal_to_nnunet_loss(base_loss, focal_loss, self.focal_weight)

    def on_train_start(self):
        result = super().on_train_start()
        if self.local_rank == 0:
            self.logger.log_summary("loss/name", "dice_ce_plus_focal")
            self.logger.log_summary("loss/formula", "L_Dice + L_CE + 0.5 * L_Focal")
            self.logger.log_summary("loss/focal_alpha", self.focal_alpha)
            self.logger.log_summary("loss/focal_gamma", self.focal_gamma)
            self.logger.log_summary("loss/focal_weight", self.focal_weight)
            self.logger.log_summary(
                "loss/deep_supervision", bool(self.enable_deep_supervision)
            )
            self.print_to_log_file(
                "Focal-added loss enabled:",
                "L_total=L_Dice+L_CE+0.5*L_Focal;",
                f"alpha={self.focal_alpha}; gamma={self.focal_gamma};",
                f"deep_supervision={self.enable_deep_supervision}",
                also_print_to_console=True,
            )
        return result
