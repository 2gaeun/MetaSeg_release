"""BrainMetaSeg trainer with per-epoch checkpoints and lesion evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import autocast
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from brainmetaseg.evaluation.lesion_metrics import compute_case_metrics, evaluate_lesion_metrics
from brainmetaseg.nnunet_ext.focal_loss import (
    get_tracked_loss_components,
    reset_loss_component_tracking,
)
from brainmetaseg.visualization.input_qc import (
    INPUT_QC_LAYOUT_VERSION,
    augmented_batch_statistics,
    create_augmented_batch_qc,
    create_preprocessed_input_qc,
    select_input_qc_cases,
)
from brainmetaseg.visualization.overlays import create_validation_overlays, select_overlay_cases


def _allow_wandb_jobid_config_change() -> None:
    """Let nnU-Net resume a W&B run from a new Slurm job ID.

    nnU-Net's built-in WandbLogger writes {"JobID": current_slurm_job_id}
    during construction. When resuming a run created by an older Slurm job,
    W&B treats that as an immutable config change and aborts unless
    allow_val_change=True is passed. Keep this patch intentionally narrow:
    only updates containing the "JobID" config key are relaxed.
    """
    if os.environ.get("BRAINMETASEG_ALLOW_WANDB_JOBID_CHANGE", "1").lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    try:
        from wandb.sdk import wandb_config
    except Exception:
        return

    config_cls = getattr(wandb_config, "Config", None)
    if config_cls is None:
        return
    original_update = getattr(config_cls, "update", None)
    if original_update is None or getattr(original_update, "_brainmetaseg_jobid_patch", False):
        return

    allow_resume_config_change = os.environ.get(
        "BRAINMETASEG_ALLOW_WANDB_RESUME_CONFIG_CHANGE", "0"
    ).lower() in {"1", "true", "yes", "on"}

    def update_allowing_jobid_change(self, d=None, allow_val_change=None, *args, **kwargs):
        if (
            isinstance(d, dict)
            and ("JobID" in d or allow_resume_config_change)
            and allow_val_change is None
        ):
            allow_val_change = True
        return original_update(self, d, allow_val_change=allow_val_change, *args, **kwargs)

    update_allowing_jobid_change._brainmetaseg_jobid_patch = True
    config_cls.update = update_allowing_jobid_change


class nnUNetTrainerBrainMeta(nnUNetTrainer):
    """Use nnU-Net defaults plus BrainMetaSeg persistence/evaluation policy."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        _allow_wandb_jobid_config_change()
        super().__init__(plans, configuration, fold, dataset_json, device)
        requested_batch_size = os.environ.get("BRAINMETASEG_BATCH_SIZE")
        if requested_batch_size is not None:
            effective_batch_size = int(requested_batch_size)
            if effective_batch_size < 1:
                raise ValueError("BRAINMETASEG_BATCH_SIZE must be at least 1")
            self.batch_size = effective_batch_size
            self.configuration_manager.configuration["batch_size"] = effective_batch_size
        self.grad_accum_steps = int(os.environ.get("BRAINMETASEG_GRAD_ACCUM_STEPS", "1"))
        if self.grad_accum_steps < 1:
            raise ValueError("BRAINMETASEG_GRAD_ACCUM_STEPS must be at least 1")
        self._grad_accum_step = 0
        self._grad_accum_has_pending = False
        if self.grad_accum_steps > 1:
            self.num_iterations_per_epoch *= self.grad_accum_steps
        # This overwrites checkpoint_latest.pth each epoch. Best and final
        # checkpoints remain separate, avoiding 1000 full checkpoint copies.
        self.save_every = 1
        enabled = os.environ.get("BRAINMETASEG_BATCH_QC_ENABLED", "1").strip().lower()
        if enabled not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError(f"invalid BRAINMETASEG_BATCH_QC_ENABLED={enabled!r}")
        self.train_batch_qc_enabled = enabled in {"1", "true", "yes", "on"}
        self.train_batch_qc_every_epochs = int(
            os.environ.get("BRAINMETASEG_BATCH_QC_EVERY_EPOCHS", "50")
        )
        self.train_batch_qc_num_samples = int(
            os.environ.get("BRAINMETASEG_BATCH_QC_NUM_SAMPLES", "2")
        )
        self.train_batch_qc_display_window = (
            float(os.environ.get("BRAINMETASEG_BATCH_QC_WINDOW_MIN", "-3.0")),
            float(os.environ.get("BRAINMETASEG_BATCH_QC_WINDOW_MAX", "3.0")),
        )
        if self.train_batch_qc_every_epochs <= 0:
            raise ValueError("BRAINMETASEG_BATCH_QC_EVERY_EPOCHS must be positive")
        if self.train_batch_qc_num_samples <= 0:
            raise ValueError("BRAINMETASEG_BATCH_QC_NUM_SAMPLES must be positive")
        if self.train_batch_qc_display_window[1] <= self.train_batch_qc_display_window[0]:
            raise ValueError("batch QC display window max must be greater than min")
        self._train_batch_qc_logged_epoch: int | None = None
        self._latest_val_patch_metrics: dict[str, float | int] = {}
        self._latest_train_loss_components: dict[str, float] = {}
        self._latest_val_loss_components: dict[str, float] = {}

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            self.logger.log_summary("training/micro_batch_size", int(self.batch_size))
            self.logger.log_summary("training/grad_accum_steps", int(self.grad_accum_steps))
            self.logger.log_summary(
                "training/effective_batch_size",
                int(self.batch_size) * int(self.grad_accum_steps),
            )
            self.logger.log_summary("training/micro_iterations_per_epoch", int(self.num_iterations_per_epoch))
            self._define_wandb_epoch_metrics()
            self.print_to_log_file(
                f"Effective training batch size: {self.batch_size * self.grad_accum_steps} "
                f"(micro_batch={self.batch_size}, accumulation={self.grad_accum_steps})",
                also_print_to_console=True,
            )
            self._create_input_qc_once()
            self.print_to_log_file(
                "Augmented train batch QC:",
                f"enabled={self.train_batch_qc_enabled}",
                f"every_epochs={self.train_batch_qc_every_epochs}",
                f"max_samples={self.train_batch_qc_num_samples}",
                f"fixed_window={self.train_batch_qc_display_window}",
                also_print_to_console=True,
            )
        if self.is_ddp:
            dist.barrier()

    def _create_input_qc_once(self):
        output_path = join(self.output_folder, "input_qc_preprocessed.png")
        layout_marker = Path(f"{output_path}.layout-version")
        existing_panels = sorted(
            Path(self.output_folder).glob("input_qc_preprocessed_sample_*.png")
        )
        current_layout = (
            layout_marker.read_text(encoding="utf-8").strip()
            if layout_marker.is_file()
            else None
        )
        if existing_panels and current_layout == INPUT_QC_LAYOUT_VERSION:
            self.logger.log_summary(
                "input_qc/local_paths",
                ", ".join(str(path) for path in existing_panels),
            )
            for panel_index, panel_path in enumerate(existing_panels, start=1):
                self._log_image_to_wandb(
                    f"input_qc/sample_{panel_index:02d}",
                    panel_path,
                    f"Existing preprocessed input QC sample {panel_index}",
                    step=self.current_epoch,
                    commit=False,
                )
            self.print_to_log_file(
                f"Input QC panels already exist, skipping: {existing_panels}"
            )
            return

        manifest_path = os.environ.get("BRAINMETASEG_EXPORT_MANIFEST")
        if not manifest_path or not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                "BRAINMETASEG_EXPORT_MANIFEST is required for channel-aware input QC: "
                f"{manifest_path!r}"
            )
        train_identifiers, _ = self.do_split()
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        selected = select_input_qc_cases(
            manifest.to_dict(orient="records"),
            train_identifiers,
            max_cases=3,
        )
        if not selected:
            raise RuntimeError(f"no training cases available for input QC in {manifest_path}")

        dataset_train, _ = self.get_tr_and_val_datasets()
        samples = []
        for record in selected:
            case_id = str(record["nnunet_case_id"])
            data, segmentation, _, _ = dataset_train.load_case(case_id)
            samples.append(
                {
                    "case_id": case_id,
                    "ch0_type": record.get("ch0_type", ""),
                    "ch1_type": record.get("ch1_type", ""),
                    "data": data[:],
                    "segmentation": segmentation[:],
                    "spacing": self.configuration_manager.spacing,
                }
            )
        qc_paths = create_preprocessed_input_qc(samples, output_path)
        layout_marker.write_text(f"{INPUT_QC_LAYOUT_VERSION}\n", encoding="utf-8")
        case_summary = ", ".join(
            f"{record['nnunet_case_id']}[{record.get('ch0_type', '')},{record.get('ch1_type', '')}]"
            for record in selected
        )
        self.logger.log_summary("input_qc/cases", case_summary)
        self.logger.log_summary(
            "input_qc/local_paths",
            ", ".join(str(path) for path in qc_paths),
        )
        for panel_index, (record, qc_path) in enumerate(
            zip(selected, qc_paths),
            start=1,
        ):
            caption = (
                f"{record['nnunet_case_id']} | "
                f"ch0={record.get('ch0_type', '')}, ch1={record.get('ch1_type', '')}"
            )
            self._log_image_to_wandb(
                f"input_qc/sample_{panel_index:02d}",
                qc_path,
                caption,
                step=self.current_epoch,
                commit=False,
            )
        self.print_to_log_file(
            f"Saved preprocessed input QC panels: {qc_paths}",
            also_print_to_console=True,
        )

    def _maybe_log_augmented_train_batch(self, batch: dict) -> None:
        """Log the first post-augmentation batch at a low epoch frequency."""
        epoch = int(self.current_epoch)
        if (
            not self.train_batch_qc_enabled
            or self.local_rank != 0
            or epoch % self.train_batch_qc_every_epochs != 0
            or self._train_batch_qc_logged_epoch == epoch
        ):
            return
        self._train_batch_qc_logged_epoch = epoch

        try:
            data = batch["data"]
            if torch.is_tensor(data):
                data_np = data.detach().float().cpu().numpy()
            else:
                data_np = np.asarray(data, dtype=np.float32)

            target = batch["target"]
            if isinstance(target, (list, tuple)):
                target = target[0]
            if torch.is_tensor(target):
                target_np = target.detach().cpu().numpy()
            else:
                target_np = np.asarray(target)

            if target_np.ndim == data_np.ndim:
                if self.label_manager.has_regions:
                    region_channel_count = target_np.shape[1] - int(
                        self.label_manager.has_ignore_label
                    )
                    reference = (target_np[:, :region_channel_count] > 0).any(axis=1)
                    if self.label_manager.has_ignore_label:
                        reference &= ~target_np[:, -1].astype(bool)
                else:
                    labels = target_np[:, 0]
                    reference = labels > 0
                    if self.label_manager.has_ignore_label:
                        reference &= labels != self.label_manager.ignore_label
            elif target_np.ndim == data_np.ndim - 1:
                reference = target_np > 0
            else:
                raise ValueError(
                    f"unexpected train target shape {target_np.shape} for data {data_np.shape}"
                )
            segmentation_np = np.asarray(reference, dtype=np.uint8)

            raw_keys = batch.get("keys", ())
            if isinstance(raw_keys, str):
                case_ids = [raw_keys]
            else:
                case_ids = [str(value) for value in raw_keys]

            qc_paths = create_augmented_batch_qc(
                data_np,
                segmentation_np,
                join(self.output_folder, "train_batch_qc"),
                epoch=epoch,
                spacing=self.configuration_manager.spacing,
                case_ids=case_ids,
                max_samples=self.train_batch_qc_num_samples,
                display_window=self.train_batch_qc_display_window,
            )
            statistics = augmented_batch_statistics(data_np, segmentation_np)
            self._log_train_batch_qc_to_wandb(
                statistics,
                qc_paths,
                case_ids,
                step=epoch,
            )
            self.print_to_log_file(
                f"Saved augmented train batch QC: {qc_paths}",
                f"statistics={statistics}",
                also_print_to_console=True,
            )
        except Exception as error:
            self.print_to_log_file(
                f"WARNING: augmented train batch QC failed without stopping training: {error}",
                also_print_to_console=True,
            )

    def train_step(self, batch: dict) -> dict:
        self._maybe_log_augmented_train_batch(batch)

        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        elif isinstance(target, tuple):
            target = tuple(item.to(self.device, non_blocking=True) for item in target)
        else:
            target = target.to(self.device, non_blocking=True)

        if self._grad_accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with context:
            output = self.network(data)
            reset_loss_component_tracking(self.loss)
            loss = self.loss(output, target)
            loss_components = get_tracked_loss_components(self.loss)
            loss_for_backward = loss / self.grad_accum_steps

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        self._grad_accum_step += 1
        self._grad_accum_has_pending = True
        if self._grad_accum_step >= self.grad_accum_steps:
            self._step_accumulated_gradients()

        result = {"loss": loss.detach().cpu().numpy()}
        result.update(
            {
                f"loss_component_{name}": value.detach().cpu().numpy()
                for name, value in loss_components.items()
            }
        )
        return result

    def _step_accumulated_gradients(self):
        if not self._grad_accum_has_pending:
            return
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        self._grad_accum_step = 0
        self._grad_accum_has_pending = False

    def on_train_epoch_end(self, train_outputs):
        if self._grad_accum_has_pending:
            self._step_accumulated_gradients()
        self._latest_train_loss_components = self._mean_loss_components(train_outputs)
        if self.local_rank == 0 and self._latest_train_loss_components:
            self.print_to_log_file(
                "Train loss components:",
                self._latest_train_loss_components,
                also_print_to_console=True,
            )
        return super().on_train_epoch_end(train_outputs)

    def validation_step(self, batch: dict) -> dict:
        """Run the native validation step and retain its prediction for patch metrics."""

        captured_output: dict[str, object] = {}

        def capture_prediction(_module, _inputs, output):
            captured_output["prediction"] = output

        handle = self.network.register_forward_hook(capture_prediction)
        try:
            reset_loss_component_tracking(self.loss)
            result = super().validation_step(batch)
        finally:
            handle.remove()

        loss_components = get_tracked_loss_components(self.loss)
        result.update(
            {
                f"loss_component_{name}": value.detach().cpu().numpy()
                for name, value in loss_components.items()
            }
        )

        output = captured_output.get("prediction")
        if output is None:
            raise RuntimeError("validation forward hook did not capture a prediction")
        if self.enable_deep_supervision:
            output = output[0]

        target = batch["target"]
        if isinstance(target, list):
            target = target[0]

        with torch.no_grad():
            if self.label_manager.has_regions:
                prediction = (torch.sigmoid(output) > 0.5).any(dim=1)
                reference = target[:, :-1] if self.label_manager.has_ignore_label else target
                reference = reference.bool().any(dim=1)
                if self.label_manager.has_ignore_label:
                    valid_mask = ~target[:, -1].bool()
                    prediction = prediction & valid_mask.to(prediction.device)
                    reference = reference & valid_mask
            else:
                prediction = output.argmax(dim=1) > 0
                reference = target[:, 0]
                if self.label_manager.has_ignore_label:
                    valid_mask = reference != self.label_manager.ignore_label
                    prediction = prediction & valid_mask.to(prediction.device)
                    reference = (reference > 0) & valid_mask
                else:
                    reference = reference > 0

            prediction_cpu = prediction.detach().cpu().numpy()
            reference_cpu = reference.detach().cpu().numpy()

        small_lesion_mm = float(os.environ.get("BRAINMETASEG_SMALL_LESION_MM", "10.0"))
        spacing = tuple(float(value) for value in self.configuration_manager.spacing)
        patch_metrics = [
            compute_case_metrics(
                reference_cpu[index],
                prediction_cpu[index],
                spacing,
                small_lesion_max_diameter_mm=small_lesion_mm,
                connectivity=26,
            )
            for index in range(reference_cpu.shape[0])
        ]
        result.update(
            patch_gt_lesions=sum(metric["gt_lesions"] for metric in patch_metrics),
            patch_predicted_lesions=sum(metric["predicted_lesions"] for metric in patch_metrics),
            patch_detected_gt_lesions=sum(
                metric["detected_gt_lesions"] for metric in patch_metrics
            ),
            patch_false_positive_lesions=sum(
                metric["false_positive_lesions"] for metric in patch_metrics
            ),
            patch_small_gt_lesions=sum(metric["small_gt_lesions"] for metric in patch_metrics),
            patch_detected_small_gt_lesions=sum(
                metric["detected_small_gt_lesions"] for metric in patch_metrics
            ),
            patch_hd95_mm=[metric["hd95_mm"] for metric in patch_metrics],
        )
        return result

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        collated = collate_outputs(val_outputs)
        self._latest_val_loss_components = self._mean_loss_components(val_outputs)
        if self.local_rank == 0 and self._latest_val_loss_components:
            self.print_to_log_file(
                "Validation loss components:",
                self._latest_val_loss_components,
                also_print_to_console=True,
            )
        aggregate = {
            "gt_lesions": int(sum(collated["patch_gt_lesions"])),
            "predicted_lesions": int(sum(collated["patch_predicted_lesions"])),
            "detected_gt_lesions": int(sum(collated["patch_detected_gt_lesions"])),
            "false_positive_lesions": int(sum(collated["patch_false_positive_lesions"])),
            "small_gt_lesions": int(sum(collated["patch_small_gt_lesions"])),
            "detected_small_gt_lesions": int(
                sum(collated["patch_detected_small_gt_lesions"])
            ),
            "hd95_mm": list(collated["patch_hd95_mm"]),
        }

        if self.is_ddp:
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, aggregate)
            aggregate = {
                "gt_lesions": sum(item["gt_lesions"] for item in gathered),
                "predicted_lesions": sum(item["predicted_lesions"] for item in gathered),
                "detected_gt_lesions": sum(item["detected_gt_lesions"] for item in gathered),
                "false_positive_lesions": sum(
                    item["false_positive_lesions"] for item in gathered
                ),
                "small_gt_lesions": sum(item["small_gt_lesions"] for item in gathered),
                "detected_small_gt_lesions": sum(
                    item["detected_small_gt_lesions"] for item in gathered
                ),
                "hd95_mm": [
                    value for item in gathered for value in item["hd95_mm"]
                ],
            }

        if self.local_rank != 0:
            return

        gt_lesions = aggregate["gt_lesions"]
        small_gt_lesions = aggregate["small_gt_lesions"]
        hd95_values = aggregate["hd95_mm"]
        num_patches = len(hd95_values)
        history = {
            "val_patch/lesion_wise_sensitivity": (
                aggregate["detected_gt_lesions"] / gt_lesions if gt_lesions else 0.0
            ),
            "val_patch/fp_per_patch": (
                aggregate["false_positive_lesions"] / num_patches if num_patches else 0.0
            ),
            "val_patch/small_lesion_sensitivity": (
                aggregate["detected_small_gt_lesions"] / small_gt_lesions
                if small_gt_lesions
                else 0.0
            ),
            "val_patch/hd95_mm_mean": float(np.mean(hd95_values)) if hd95_values else 0.0,
            "val_patch/hd95_mm_median": (
                float(np.median(hd95_values)) if hd95_values else 0.0
            ),
            "val_patch/gt_lesions": gt_lesions,
            "val_patch/predicted_lesions": aggregate["predicted_lesions"],
        }
        self._latest_val_patch_metrics = dict(history)
        self.print_to_log_file(
            "Patch validation lesion metrics:",
            history,
            also_print_to_console=True,
        )

    def on_epoch_end(self):
        epoch = int(self.current_epoch)
        super().on_epoch_end()
        if self.local_rank == 0:
            self._save_epoch_checkpoint(epoch)
            self._log_consolidated_epoch_to_wandb(epoch)

    def _save_epoch_checkpoint(self, epoch: int):
        if epoch == self.num_epochs - 1:
            return
        checkpoint_path = join(self.output_folder, f"checkpoint_epoch_{epoch:04d}.pth")
        if os.path.isfile(checkpoint_path):
            return
        self.save_checkpoint(checkpoint_path)
        self.print_to_log_file(
            f"Saved retained epoch checkpoint: {checkpoint_path}",
            also_print_to_console=True,
        )

    def perform_actual_validation(self, save_probabilities: bool = False):
        super().perform_actual_validation(save_probabilities)
        if self.local_rank != 0:
            return

        small_lesion_mm = float(os.environ.get("BRAINMETASEG_SMALL_LESION_MM", "10.0"))
        if small_lesion_mm <= 0:
            raise ValueError("BRAINMETASEG_SMALL_LESION_MM must be positive")

        validation_folder = join(self.output_folder, "validation")
        result = evaluate_lesion_metrics(
            reference_dir=join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
            prediction_dir=validation_folder,
            output_file=join(validation_folder, "lesion_metrics.json"),
            file_ending=self.dataset_json["file_ending"],
            small_lesion_max_diameter_mm=small_lesion_mm,
            connectivity=26,
        )
        summary = result["summary"]
        for key in (
            "lesion_wise_sensitivity",
            "fp_per_case",
            "small_lesion_sensitivity",
            "hd95_mm_mean",
            "hd95_mm_median",
            "gt_lesions",
            "predicted_lesions",
        ):
            value = summary[key]
            if value is not None:
                self.logger.log_summary(f"final_val/{key}", value)

        self.print_to_log_file(
            "BrainMetaSeg lesion metrics:",
            summary,
            also_print_to_console=True,
        )

        overlay_cases = select_overlay_cases(result["per_case"], max_cases=3)
        overlay_path = create_validation_overlays(
            reference_dir=join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
            prediction_dir=validation_folder,
            image_dir=join(os.environ["nnUNet_raw"], self.plans_manager.dataset_name, "imagesTr"),
            case_filenames=overlay_cases,
            output_file=join(validation_folder, "validation_overlays.png"),
            file_ending=self.dataset_json["file_ending"],
            per_case_metrics=result["per_case"],
        )
        self.logger.log_summary("final_val/overlay_cases", ", ".join(overlay_cases))
        self.logger.log_summary("final_val/overlay_path", str(overlay_path))
        self._log_image_to_wandb(
            "final_val/segmentation_overlays",
            overlay_path,
            f"Cases: {', '.join(overlay_cases)}",
            step=self.current_epoch,
            commit=True,
        )

    def _log_train_batch_qc_to_wandb(
        self,
        statistics,
        image_paths,
        case_ids,
        *,
        step,
    ):
        try:
            import wandb

            if wandb.run is not None:
                payload = dict(statistics)
                for panel_index, image_path in enumerate(image_paths, start=1):
                    case_id = (
                        case_ids[panel_index - 1]
                        if panel_index <= len(case_ids)
                        else f"batch_sample_{panel_index:02d}"
                    )
                    payload[f"train_batch_qc/sample_{panel_index:02d}"] = wandb.Image(
                        str(image_path),
                        caption=f"epoch={step}, case={case_id}, post-augmentation train batch",
                    )
                wandb.run.log(payload, step=step)
        except Exception as error:
            self.print_to_log_file(
                f"WARNING: train batch QC was saved locally but W&B upload failed: {error}",
                also_print_to_console=True,
            )


    def _define_wandb_epoch_metrics(self):
        try:
            import wandb

            if wandb.run is not None:
                wandb.define_metric("epoch")
                wandb.define_metric("epoch/*", step_metric="epoch")
                wandb.define_metric("val_patch/*", step_metric="epoch")
        except Exception as error:
            self.print_to_log_file(
                f"WARNING: W&B epoch metric definitions failed: {error}",
                also_print_to_console=True,
            )

    def _log_consolidated_epoch_to_wandb(self, epoch: int):
        try:
            import wandb

            if wandb.run is None:
                return

            start_time = self.logger.get_value("epoch_start_timestamps", step=epoch)
            end_time = self.logger.get_value("epoch_end_timestamps", step=epoch)
            payload = {
                "epoch": epoch,
                "epoch/train_loss": self._wandb_scalar(
                    self.logger.get_value("train_losses", step=epoch)
                ),
                "epoch/val_loss": self._wandb_scalar(
                    self.logger.get_value("val_losses", step=epoch)
                ),
                "epoch/lr": self._wandb_scalar(self.logger.get_value("lrs", step=epoch)),
                "epoch/mean_fg_dice": self._wandb_scalar(
                    self.logger.get_value("mean_fg_dice", step=epoch)
                ),
                "epoch/ema_fg_dice": self._wandb_scalar(
                    self.logger.get_value("ema_fg_dice", step=epoch)
                ),
                "epoch/duration_s": self._wandb_scalar(end_time - start_time),
            }
            payload.update(
                {
                    key: self._wandb_scalar(value)
                    for key, value in self._latest_val_patch_metrics.items()
                }
            )
            payload.update(
                {
                    f"epoch/train_loss_{name}": self._wandb_scalar(value)
                    for name, value in self._latest_train_loss_components.items()
                }
            )
            payload.update(
                {
                    f"epoch/val_loss_{name}": self._wandb_scalar(value)
                    for name, value in self._latest_val_loss_components.items()
                }
            )
            wandb.run.log(payload)
        except Exception as error:
            self.print_to_log_file(
                f"WARNING: consolidated epoch metrics were computed but W&B upload failed: {error}",
                also_print_to_console=True,
            )


    @staticmethod
    def _wandb_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    
    @staticmethod
    def _mean_loss_components(outputs):
        if not outputs:
            return {}
        component_keys = sorted(
            key for key in outputs[0] if key.startswith("loss_component_")
        )
        return {
            key.removeprefix("loss_component_"): float(
                np.mean([np.asarray(output[key]).item() for output in outputs])
            )
            for key in component_keys
        }

    def _log_image_to_wandb(self, key, image_path, caption, step, commit):
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.log(
                    {key: wandb.Image(str(image_path), caption=caption)},
                    step=step,
                    commit=commit,
                )
        except Exception as error:
            self.print_to_log_file(
                f"WARNING: image was saved locally but W&B upload failed: {error}",
                also_print_to_console=True,
            )

    def _log_metrics_to_wandb(self, metrics, step):
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.log(metrics, step=step)
        except Exception as error:
            self.print_to_log_file(
                f"WARNING: patch metrics were computed but W&B upload failed: {error}",
                also_print_to_console=True,
            )
