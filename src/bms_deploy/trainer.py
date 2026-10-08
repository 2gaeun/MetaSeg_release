"""Inference-only adapter for the bundled Focal checkpoint.

The research Components -> Focal -> BrainMeta -> nnUNetTrainer inheritance
chain does not override build_network_architecture. Focal loss and logging are
training-only; original source files are retained in weights/*/trainer_source.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerBrainMetaFocalLR3e3Components(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        raise RuntimeError('This portable trainer adapter supports network construction for inference only')
