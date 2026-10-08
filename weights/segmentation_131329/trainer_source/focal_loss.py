"""Focal augmentation for nnU-Net's existing compound segmentation loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BrainMetaFocalLoss(nn.Module):
    """Alpha-balanced focal loss for label-map or region targets."""

    def __init__(self, alpha=0.25, gamma=2.0, has_regions=False, ignore_label=None):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.has_regions = bool(has_regions)
        self.ignore_label = ignore_label

    def forward(self, logits, target):
        if self.has_regions:
            return self._region_focal(logits, target)
        return self._label_focal(logits, target)

    def _label_focal(self, logits, target):
        labels = target[:, 0] if target.ndim == logits.ndim else target
        labels = labels.long()
        ignore_index = -100 if self.ignore_label is None else int(self.ignore_label)
        ce = F.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)
        valid = labels != ignore_index
        pt = torch.exp(-ce)
        alpha_t = torch.where(
            labels > 0,
            ce.new_tensor(self.alpha),
            ce.new_tensor(1.0 - self.alpha),
        )
        focal = alpha_t * (1.0 - pt).pow(self.gamma) * ce
        return focal[valid].sum() / valid.sum().clamp_min(1)

    def _region_focal(self, logits, target):
        regions = target[:, : logits.shape[1]].to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, regions, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = probabilities * regions + (1.0 - probabilities) * (1.0 - regions)
        alpha_t = self.alpha * regions + (1.0 - self.alpha) * (1.0 - regions)
        focal = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        if self.ignore_label is None:
            return focal.mean()
        valid = (~target[:, -1:].bool()).expand_as(focal)
        return focal[valid].sum() / valid.sum().clamp_min(1)


class DiceCEFocalLoss(nn.Module):
    """Reuse nnU-Net's Dice+CE loss and add a weighted focal term."""

    def __init__(self, base_loss, focal_loss, focal_weight=0.5):
        super().__init__()
        if focal_weight < 0:
            raise ValueError("focal_weight must be non-negative")
        self.base_loss = base_loss
        self.focal_loss = focal_loss
        self.focal_weight = float(focal_weight)
        self._component_records = []

    def forward(self, logits, target):
        dice, ce = self._base_components(logits, target)
        focal = self.focal_loss(logits, target)
        focal_weighted = self.focal_weight * focal
        total = dice + ce + focal_weighted
        self._component_records.append(
            {
                "dice": dice.detach(),
                "ce": ce.detach(),
                "focal": focal.detach(),
                "focal_weighted": focal_weighted.detach(),
                "total": total.detach(),
            }
        )
        return total

    def reset_component_tracking(self):
        self._component_records.clear()

    def get_component_records(self):
        return tuple(self._component_records)

    def _base_components(self, logits, target):
        """Evaluate nnU-Net Dice and CE/BCE terms without changing their math."""
        zero = logits.sum() * 0.0
        if all(
            hasattr(self.base_loss, name)
            for name in ("dc", "ce", "weight_dice", "weight_ce")
        ):
            if hasattr(self.base_loss, "ignore_label"):
                ignore_label = self.base_loss.ignore_label
                if ignore_label is not None:
                    mask = target != ignore_label
                    target_dice = torch.where(mask, target, 0)
                    num_valid = mask.sum()
                else:
                    mask = None
                    target_dice = target
                    num_valid = None
                dice = (
                    self.base_loss.weight_dice
                    * self.base_loss.dc(logits, target_dice, loss_mask=mask)
                    if self.base_loss.weight_dice != 0
                    else zero
                )
                ce = (
                    self.base_loss.weight_ce * self.base_loss.ce(logits, target[:, 0])
                    if self.base_loss.weight_ce != 0
                    and (ignore_label is None or num_valid > 0)
                    else zero
                )
                return dice, ce

            if hasattr(self.base_loss, "use_ignore_label"):
                if self.base_loss.use_ignore_label:
                    mask = (
                        ~target[:, -1:]
                        if target.dtype == torch.bool
                        else (1 - target[:, -1:]).bool()
                    )
                    target_regions = target[:, :-1]
                else:
                    mask = None
                    target_regions = target
                dice = self.base_loss.weight_dice * self.base_loss.dc(
                    logits, target_regions, loss_mask=mask
                )
                target_regions = target_regions.float()
                if mask is None:
                    ce_raw = self.base_loss.ce(logits, target_regions)
                else:
                    ce_raw = (self.base_loss.ce(logits, target_regions) * mask).sum()
                    ce_raw = ce_raw / torch.clamp(mask.sum(), min=1e-8)
                return dice, self.base_loss.weight_ce * ce_raw

        # Preserve the total formula for test doubles and future compound losses.
        return self.base_loss(logits, target), zero


def add_focal_to_nnunet_loss(base_loss, focal_loss, focal_weight=0.5):
    """Preserve nnU-Net's deep-supervision wrapper and exact weight factors."""
    if base_loss.__class__.__name__ == "DeepSupervisionWrapper":
        return base_loss.__class__(
            DiceCEFocalLoss(base_loss.loss, focal_loss, focal_weight),
            base_loss.weight_factors,
        )
    return DiceCEFocalLoss(base_loss, focal_loss, focal_weight)


def _tracked_focal_loss(loss):
    if isinstance(loss, DiceCEFocalLoss):
        return loss
    inner = getattr(loss, "loss", None)
    return inner if isinstance(inner, DiceCEFocalLoss) else None


def reset_loss_component_tracking(loss):
    tracked = _tracked_focal_loss(loss)
    if tracked is not None:
        tracked.reset_component_tracking()


def get_tracked_loss_components(loss):
    """Return deep-supervision-weighted components from the latest loss call."""
    tracked = _tracked_focal_loss(loss)
    if tracked is None:
        return {}
    records = tracked.get_component_records()
    if not records:
        return {}
    if tracked is loss:
        weights = (1.0,)
    else:
        weights = tuple(weight for weight in loss.weight_factors if weight != 0)
    if len(records) != len(weights):
        raise RuntimeError(
            "loss component tracking mismatch: "
            f"records={len(records)}, deep_supervision_weights={len(weights)}"
        )
    return {
        name: sum(weight * record[name] for weight, record in zip(weights, records))
        for name in records[0]
    }
