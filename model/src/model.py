import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from typing import List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
# Metrics
from torchmetrics import MetricCollection, Precision, Recall
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.average_precision import AveragePrecision  # AUPR
# from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
# from torchmetrics.classification.cohen_kappa import CohenKappa
from torchmetrics.classification.confusion_matrix import ConfusionMatrix

from transformers import AutoConfig, AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup
import torchvision.models as tv_models

from src.model_catalog import MODEL_DICT  # keep the folder name
from src.loss import SoftTargetCrossEntropy
from src.mixup import Mixup
from utils.plot import plot_confusion_matrix, plot_ROC_PR_curves, calculate_AUROC_AUPR



class ClassificationModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str = "vit-b16-224-dino-v2",
            model_path: Optional[str] = None,
            optimizer: str = "adamw",
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.999),
            momentum: float = 0.9,
            weight_decay: float = 0.01,
            scheduler: str = "cosine",
            warmup_steps: int = 100,
            n_classes: int = 10,
            mixup_alpha: float = 0.0,
            cutmix_alpha: float = 0.0,
            mix_prob: float = 1.0,
            label_smoothing: float = 0.0,
            image_size: int = 224,
            weights: Optional[str] = None,
            training_mode: str = "full",
            lora_r: int = 16,
            lora_alpha: int = 16,
            lora_target_modules: Optional[List[str]] = None,
            lora_dropout: float = 0.0,
            lora_bias: str = "none",
            from_scratch: bool = False,
            class_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.class_labels = class_names  # Save class_labels for plotting
        print(f"Model initialized with class_labels: {self.class_labels}")

        self.save_hyperparameters()

        self.hparams.class_names = class_names  # Save class_names in hparams for logging
        # print(f"Model initialized with class_names: {self.hparams.class_names}")

        # --- Network Initialization ---
        try:
            model_identifier = MODEL_DICT[self.hparams.model_name]
        except KeyError:
            raise ValueError(f"Model '{self.hparams.model_name}' is not available.")

        is_torchvision_model = model_identifier.startswith("torchvision/")
        is_hf_model = not is_torchvision_model

        if is_torchvision_model:
            tv_model_key = model_identifier.split('/')[-1]
            print(f"Loading Torchvision model: {tv_model_key} for {self.hparams.n_classes} classes.")
            if self.hparams.from_scratch:
                self.net = getattr(tv_models, tv_model_key)(weights=None, num_classes=self.hparams.n_classes)
            else:  # Load pretrained
                try:
                    weights_enum_name = f"{tv_model_key.upper()}_Weights"
                    if hasattr(tv_models, weights_enum_name):
                        weights_enum = getattr(tv_models, weights_enum_name).DEFAULT
                        self.net = getattr(tv_models, tv_model_key)(weights=weights_enum)
                    else:
                        self.net = getattr(tv_models, tv_model_key)(pretrained=True)  # type: ignore
                except Exception as e:
                    print(
                        f"Warning: Failed to load Torchvision pretrained weights for {tv_model_key}. Error: {e}. Initializing from scratch.")
                    self.net = getattr(tv_models, tv_model_key)(weights=None, num_classes=self.hparams.n_classes)

                # Replace classifier head
                if hasattr(self.net, 'classifier') and isinstance(self.net.classifier, (nn.Linear, nn.Sequential)):
                    if isinstance(self.net.classifier, nn.Linear):
                        num_ftrs = self.net.classifier.in_features
                        self.net.classifier = nn.Linear(num_ftrs, self.hparams.n_classes)
                    elif isinstance(self.net.classifier, nn.Sequential):
                        final_linear_layer_idx = -1
                        for idx, layer in reversed(list(enumerate(self.net.classifier))):  # type: ignore
                            if isinstance(layer, nn.Linear): final_linear_layer_idx = idx; break
                        if final_linear_layer_idx != -1:
                            num_ftrs = self.net.classifier[final_linear_layer_idx].in_features  # type: ignore
                            self.net.classifier[final_linear_layer_idx] = nn.Linear(num_ftrs,
                                                                                    self.hparams.n_classes)  # type: ignore
                        else:
                            print(f"Warning: No Linear layer in Sequential classifier for {tv_model_key}")
                elif hasattr(self.net, 'fc') and isinstance(self.net.fc, nn.Linear):
                    num_ftrs = self.net.fc.in_features
                    self.net.fc = nn.Linear(num_ftrs, self.hparams.n_classes)
                else:
                    print(f"Warning: Could not auto-replace classifier for Torchvision model {tv_model_key}")

        elif is_hf_model:
            hf_model_id = model_identifier
            print(f"Loading Hugging Face model: {hf_model_id} for {self.hparams.n_classes} classes.")
            if self.hparams.from_scratch:
                config = AutoConfig.from_pretrained(hf_model_id, num_labels=self.hparams.n_classes,
                                                    image_size=self.hparams.image_size if 'vit' in self.hparams.model_name.lower() else None)
                self.net = AutoModelForImageClassification.from_config(config)
            else:
                # Load pretrained model from Hugging Face
                if hasattr(self.hparams, 'model_path') and self.hparams.model_path:
                    self.net = AutoModelForImageClassification.from_pretrained(
                        self.hparams.model_path,
                        num_labels=self.hparams.n_classes,
                        ignore_mismatched_sizes=True
                    )
                # Load from Hugging Face Hub
                else:
                    self.net = AutoModelForImageClassification.from_pretrained(
                        hf_model_id, num_labels=self.hparams.n_classes, ignore_mismatched_sizes=True,
                        # If using ViT/DINO, specify image size
                        image_size=self.hparams.image_size if 'vit' in self.hparams.model_name.lower()
                                                              or 'dino' in self.hparams.model_name.lower() else None)
        else:
            raise ValueError(f"Unknown model source for identifier: {model_identifier}")
            # Load the model from the local shared path


        if self.hparams.weights:
            # Checkpoint loading logic
            print(f"Loading weights from checkpoint: {self.hparams.weights}")
            try:
                ckpt = torch.load(self.hparams.weights, map_location=lambda storage, loc: storage)["state_dict"]
                new_state_dict = {k.replace("net.", "", 1) if k.startswith("net.") else k: v for k, v in ckpt.items()}
                missing, unexpected = self.net.load_state_dict(new_state_dict, strict=False)
                if missing: print(f"Ckpt load missing keys: {missing}")
                if unexpected: print(f"Ckpt load unexpected keys: {unexpected}")
            except Exception as e:
                print(f"Error loading weights from {self.hparams.weights}: {e}")

        # --- Training Mode (LoRA, Linear, Full) ---
        if self.hparams.training_mode == "linear":
            print("Linear probing: Freezing backbone, training classifier.")
            for name, param in self.net.named_parameters():
                is_classifier = ("classifier" in name) or ("fc" in name) or ("head" in name)  # Common patterns
                param.requires_grad = is_classifier
                if is_classifier: print(f"  Linear probing: Unfrozen {name}")
        elif self.hparams.training_mode == "lora":
            # Adapt if needed
            effective_lora_target_modules = self.hparams.lora_target_modules
            if effective_lora_target_modules is None:  # Basic auto-detection
                if any(vit_kw in self.hparams.model_name.lower() for vit_kw in ["vit", "deit", "beit", "dino"]):
                    effective_lora_target_modules = ["query", "value"]
                # else: print("LoRA for CNN: `lora_target_modules` not specified. Specify Conv2d layers for best results.")

            modules_to_save = [name for name, mod in self.net.named_modules() if isinstance(mod, nn.Linear) and (
                    ("classifier" in name) or ("fc" in name) or ("head" in name))]

            if effective_lora_target_modules:
                lora_config = LoraConfig(
                    r=self.hparams.lora_r, lora_alpha=self.hparams.lora_alpha,
                    target_modules=effective_lora_target_modules, lora_dropout=self.hparams.lora_dropout,
                    bias=self.hparams.lora_bias, modules_to_save=modules_to_save if modules_to_save else None
                )
                self.net = get_peft_model(self.net, lora_config)
                print(
                    f"Applied LoRA: r={self.hparams.lora_r}, targets={effective_lora_target_modules}, saved={modules_to_save}")
                if hasattr(self.net,
                           'print_trainable_parameters'): self.net.print_trainable_parameters()  # type: ignore
            else:
                print("LoRA not applied as no target modules were effectively specified.")
        elif self.hparams.training_mode == "full":
            print("Full fine-tuning: All parameters are trainable.")
            for param in self.net.parameters(): param.requires_grad = True




        # --- Metrics Initialization ---
        num_actual_classes = self.hparams.n_classes
        common_args_multiclass_macro = {"task": "multiclass", "num_classes": num_actual_classes, "average": "macro"}
        common_args_multiclass_weighted = {"task": "multiclass", "num_classes": num_actual_classes,
                                           "average": "weighted"}
        common_args_multiclass_scores = {"task": "multiclass", "num_classes": num_actual_classes}

        # Metrics for standard logging ( to metrics.csv)
        def make_metrics():
            return {
                'acc': Accuracy(**common_args_multiclass_scores, top_k=1),
                'f1_macro': F1Score(**common_args_multiclass_macro),
                'f1_weighted': F1Score(**common_args_multiclass_weighted),
                'precision_macro': Precision(**common_args_multiclass_macro),
                'precision_weighted': Precision(**common_args_multiclass_weighted),
                'recall_macro': Recall(**common_args_multiclass_macro),
                'recall_weighted': Recall(**common_args_multiclass_weighted),
                'auroc_macro': AUROC(**common_args_multiclass_macro),
                'auroc_weighted': AUROC(**common_args_multiclass_weighted),
                'aupr_macro': AveragePrecision(**common_args_multiclass_macro),
                'aupr_weighted': AveragePrecision(**common_args_multiclass_weighted),
            }

        self.train_metrics = MetricCollection(make_metrics())
        self.val_metrics = MetricCollection(make_metrics())
        self.test_metrics = MetricCollection(make_metrics())

        # metrics_dict_aggregated = {
        #     'acc': Accuracy(**common_args_multiclass_scores, top_k=1),  # Top-1 accuracy
        #     'f1_macro': F1Score(**common_args_multiclass_macro),
        #     'f1_weighted': F1Score(**common_args_multiclass_weighted),
        #     'precision_macro': Precision(**common_args_multiclass_macro),
        #     'precision_weighted': Precision(**common_args_multiclass_weighted),
        #     'recall_macro': Recall(**common_args_multiclass_macro),
        #     'recall_weighted': Recall(**common_args_multiclass_weighted),
        #     'auroc_macro': AUROC(**common_args_multiclass_macro),
        #     'auroc_weighted': AUROC(**common_args_multiclass_weighted),
        #     'aupr_macro': AveragePrecision(**common_args_multiclass_macro),
        #     'aupr_weighted': AveragePrecision(**common_args_multiclass_weighted),
        # }
        # self.train_metrics = MetricCollection(metrics_dict_aggregated.copy())
        # self.val_metrics = MetricCollection(metrics_dict_aggregated.copy())
        # self.test_metrics = MetricCollection(
        #     metrics_dict_aggregated.copy())  # For self.log in test_step

        # Specific metrics for detailed test reporting in on_test_epoch_end
        self.test_per_class_stats_calculator = StatScores(**common_args_multiclass_scores, average=None)
        self.test_conf_matrix_calculator = ConfusionMatrix(**common_args_multiclass_scores, normalize=None)
        self.test_conf_matrix_normed = ConfusionMatrix(**common_args_multiclass_scores,
                                                       normalize='true')  # normalized by true class

        # Accumulators for raw predictions/targets for sklearn metrics & plots
        self.test_all_preds_probs: List[torch.Tensor] = []
        self.test_all_targets: List[torch.Tensor] = []

        # --- Loss and Mixup ---
        self.loss_fn = SoftTargetCrossEntropy() if self.hparams.label_smoothing > 0 or self.hparams.mixup_alpha > 0 or self.hparams.cutmix_alpha > 0 else nn.CrossEntropyLoss()
        self.mixup = Mixup(mixup_alpha=self.hparams.mixup_alpha, cutmix_alpha=self.hparams.cutmix_alpha,
                           prob=self.hparams.mix_prob, label_smoothing=self.hparams.label_smoothing,
                           num_classes=self.hparams.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if "transformers" in str(self.net.__class__).lower() or "dinov2" in str(self.net.__class__).lower():
            output = self.net(pixel_values=x)
            return output.logits if hasattr(output, 'logits') else output
        else:
            return self.net(x)

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str) -> torch.Tensor:
        x, y_indices = batch
        is_train = mode == "train"

        # SYNC_DIST for logging
        SYNC_DIST = True if self.trainer and self.trainer.world_size > 1 else False

        if is_train and (self.hparams.mixup_alpha > 0 or self.hparams.cutmix_alpha > 0):
            x, y_soft_labels = self.mixup(x, y_indices)
        else:
            y_soft_labels = F.one_hot(y_indices, num_classes=self.hparams.n_classes).float() if isinstance(self.loss_fn,
                                                                                                           SoftTargetCrossEntropy) else y_indices

        pred_logits = self(x)  # Raw logits

        loss = self.loss_fn(pred_logits, y_soft_labels)
        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=is_train, prog_bar=True, sync_dist=SYNC_DIST)

        # Update and log standard aggregated metrics
        current_metrics_collection = getattr(self, f"{mode}_metrics", None)
        # if mode == "test": current_metrics_collection = self.test_metrics

        if current_metrics_collection:
            current_metrics_collection.update(pred_logits, y_indices)
            # For train/val, log them at epoch end. For test, they are logged in on_test_epoch_end from compute().
            # However, PL's printed table for test uses metrics logged in test_step with on_epoch=True.
            if mode == "test":  # Log test metrics per step for the table, will be aggregated
                for k, v in current_metrics_collection.items():  # type: ignore
                    self.log(f"test_{k}", v, on_epoch=True, on_step=False, prog_bar=False, sync_dist=SYNC_DIST)

        # Specific actions for test mode
        if mode == "test":
            self.test_per_class_stats_calculator.update(pred_logits, y_indices)
            self.test_conf_matrix_calculator.update(pred_logits, y_indices)
            self.test_conf_matrix_normed.update(pred_logits, y_indices)
            pred_probs = torch.softmax(pred_logits, dim=1)
            self.test_all_preds_probs.append(pred_probs.detach().cpu())
            self.test_all_targets.append(y_indices.detach().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")  # Loss is logged if needed, no need to return it here

    def on_train_epoch_start(self):
        if hasattr(self, 'train_metrics'): self.train_metrics.reset()

    def on_validation_epoch_start(self):
        if hasattr(self, 'val_metrics'): self.val_metrics.reset()

    def on_test_epoch_start(self):
        # Reset all test-specific accumulators and metrics
        if hasattr(self, 'test_metrics'): self.test_metrics.reset()
        if hasattr(self, 'test_per_class_stats_calculator'): self.test_per_class_stats_calculator.reset()
        if hasattr(self, 'test_conf_matrix_calculator'): self.test_conf_matrix_calculator.reset()
        if hasattr(self, 'test_conf_matrix_normed'): self.test_conf_matrix_normed.reset()
        self.test_all_preds_probs.clear()
        self.test_all_targets.clear()

    def on_train_epoch_end(self):
        if not self.trainer.sanity_checking and hasattr(self, 'train_metrics'):
            metrics_to_log = self.train_metrics.compute()
            self.log_dict({f"train_{k}": v for k, v in metrics_to_log.items()},
                          sync_dist=True if self.trainer.world_size > 1 else False)
            # self.train_metrics.reset() # Reset by on_train_epoch_start

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking and hasattr(self, 'val_metrics'):
            metrics_to_log = self.val_metrics.compute()
            self.log_dict({f"val_{k}": v for k, v in metrics_to_log.items()},
                          sync_dist=True if self.trainer.world_size > 1 else False)
            # self.val_metrics.reset() # Reset by on_validation_epoch_start

    def on_test_epoch_end(self):
        # Determine Output Directory
        if hasattr(self.trainer.logger, 'log_dir') and self.trainer.logger.log_dir:
            output_dir = self.trainer.logger.log_dir
        elif hasattr(self.trainer.logger, 'save_dir') and self.trainer.logger.save_dir:
            output_dir = os.path.join(self.trainer.logger.save_dir, self.trainer.logger.name,
                                      f"version_{self.trainer.logger.version}")
        else:
            output_dir = self.trainer.default_root_dir if self.trainer.default_root_dir else "."
        os.makedirs(output_dir, exist_ok=True)

        # --- Compute and Log Aggregated Metrics (for metrics.csv & final table) ---
        # These were updated in test_step with on_epoch=True, so PL aggregates them for the table.
        final_aggregated_metrics = self.test_metrics.compute()
        # self.test_metrics.reset() # Reset by on_test_epoch_start

        per_class_stats = self.test_per_class_stats_calculator.compute()
        # self.test_per_class_stats_calculator.reset() # Reset by on_test_epoch_start

        per_class_metrics_list = []  # List of dicts for DataFrame
        df_per_class = pd.DataFrame()  # Initialize

        if per_class_stats is not None and per_class_stats.ndim == 2 and per_class_stats.shape[
            0] == self.hparams.n_classes:
            for i in range(self.hparams.n_classes):
                tp, fp, tn, fn, sup = per_class_stats[i, :].tolist()
                accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                per_class_metrics_list.append({
                    "class_name": self.class_labels[i],
                    "support": int(sup), "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
                })
            df_per_class = pd.DataFrame(per_class_metrics_list)
        else:
            print(
                f"--- WARNING: Per-class stats not available or shape mismatch. Shape: {per_class_stats.shape if per_class_stats is not None else 'None'}")

        # --- AUROC/AUPR and Curve Plotting ---
        if not self.test_all_preds_probs or not self.test_all_targets:
            print("--- WARNING: No predictions/targets collected for AUROC/AUPR calculation. Skipping.")
            return

        try:
            y_probs_np = torch.cat(self.test_all_preds_probs).numpy()
            y_true_np = torch.cat(self.test_all_targets).numpy()

            y_true_bin = label_binarize(y_true_np, classes=list(range(self.hparams.n_classes)))
            if self.hparams.n_classes == 2 and y_true_bin.ndim == 1:
                y_true_bin = y_true_bin.reshape(-1, 1)

            # Metrics
            auroc_vals, aupr_vals = calculate_AUROC_AUPR(y_true_bin, y_probs_np,
                                                                      self.hparams.n_classes)
            if not df_per_class.empty and len(auroc_vals) == len(df_per_class):
                df_per_class["auroc"] = auroc_vals
                df_per_class["aupr"] = aupr_vals

            # Plots
            plot_ROC_PR_curves(y_true_bin, y_probs_np, self.class_labels,
                              os.path.join(output_dir, "roc_curves_per_class.png"), curve_type="roc")
            plot_ROC_PR_curves(y_true_bin, y_probs_np, self.class_labels,
                              os.path.join(output_dir, "pr_curves_per_class.png"), curve_type="pr")

        except Exception as e:
            print(f"--- ERROR during sklearn metrics/plotting: {e}")
            import traceback
            traceback.print_exc()

        # Save Metrics
        if not df_per_class.empty:
            per_class_csv = os.path.join(output_dir, "per_class_detailed_metrics.csv")
            try:
                df_per_class.to_csv(per_class_csv, index=False, float_format="%.4f")
                print(f"Saved/Updated per-class metrics to: {per_class_csv}")
            except Exception as e:
                print(f"Failed to save per-class CSV: {e}")


        # --- Confusion Matrix ---
        # Regular confusion matrix
        plot_confusion_matrix(
            self.test_conf_matrix_calculator.compute(),
            os.path.join(output_dir, "confusion_matrix_test.png"),
            self.class_labels,
            title="Confusion Matrix",
            fmt='d'
        )

        # Normalized confusion matrix
        plot_confusion_matrix(
            self.test_conf_matrix_normed.compute(),
            os.path.join(output_dir, "confusion_matrix_normed_test.png"),
            self.class_labels,
            title="Normalized Confusion Matrix",
            fmt='.2f'
        )

        # --- Save all aggregated metrics to JSON for easy parsing ---
        all_metrics_to_save_json = {}
        for k, v in final_aggregated_metrics.items():  # From self.test_metrics
            all_metrics_to_save_json[f"aggregated_{k}"] = v.item() if hasattr(v, 'item') else v

        json_path = os.path.join(output_dir, "all_test_metrics_summary.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(all_metrics_to_save_json, f, indent=4)
            print(f"Saved all test metrics summary to: {json_path}")
        except Exception as e_json:
            print(f"Failed to save metrics summary JSON: {e_json}")

    def configure_optimizers(self):
        if hasattr(self.net, 'parameters'):
            trainable_params = filter(lambda p: p.requires_grad, self.net.parameters())
        else:
            raise ValueError("self.net does not have parameters attribute")

        if self.hparams.optimizer.lower() == "adam":
            optimizer = Adam(trainable_params, lr=self.hparams.lr, betas=self.hparams.betas,
                             weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = AdamW(trainable_params, lr=self.hparams.lr, betas=self.hparams.betas,
                              weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = SGD(trainable_params, lr=self.hparams.lr, momentum=self.hparams.momentum,
                            weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Optimizer {self.hparams.optimizer} not supported.")

        num_training_steps = 0
        if self.trainer:
            if self.trainer.max_steps and self.trainer.max_steps != -1:
                num_training_steps = self.trainer.max_steps
            elif self.trainer.max_epochs and hasattr(self.trainer.datamodule,
                                                     'train_dataloader') and self.trainer.datamodule.train_dataloader():
                num_training_steps = self.trainer.max_epochs * len(
                    self.trainer.datamodule.train_dataloader()) // getattr(self.trainer, 'accumulate_grad_batches', 1)
            elif hasattr(self.trainer,
                         'estimated_stepping_batches') and self.trainer.estimated_stepping_batches is not None and self.trainer.estimated_stepping_batches != float(
                'inf'):
                num_training_steps = int(self.trainer.estimated_stepping_batches)
            else:
                print("Warning: Defaulting num_training_steps for scheduler to 100000.")
                num_training_steps = 100000
        else:
            print(
                "Warning: Trainer not available, defaulting num_training_steps for scheduler to 100000.")
            num_training_steps = 100000

        if self.hparams.scheduler.lower() == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps,
                                                        num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.scheduler.lower() == "none" or not self.hparams.scheduler:
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(f"Scheduler {self.hparams.scheduler} not supported.")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
