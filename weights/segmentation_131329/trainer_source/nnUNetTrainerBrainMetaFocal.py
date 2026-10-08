"""Standard BrainMetaSeg trainer with Dice + CE + focal loss."""

from brainmetaseg.nnunet_ext.focal_trainer_mixin import BrainMetaFocalTrainerMixin
from brainmetaseg.nnunet_ext.nnUNetTrainerBrainMeta import nnUNetTrainerBrainMeta


class nnUNetTrainerBrainMetaFocal(BrainMetaFocalTrainerMixin, nnUNetTrainerBrainMeta):
    pass
