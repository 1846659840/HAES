"""
HAES Incremental Training Loop.
Implements the complete training protocol from Section III & IV.

Training flow:
1. Phase 1: Initial training with only classification loss
2. Warm-up (3 epochs): Uniform distillation weights to avoid premature noise solidification
3. Phase p > 1: Full HAES training with all constraints active
   - Freeze previous model as Teacher
   - Apply KD, MSE, Routing-KL, EWC constraints
   - ELM structural adaptation (Add/Merge/Recycle)
4. After each phase: Update EWC Fisher & Anchor, consolidate model

Key hyperparameters (Section IV-B):
- Adam optimizer: lr=5e-4, weight_decay=1e-4, batch_size=64
- Temperature tau=4 for distillation
- Top-K=3 segments for MIL aggregation
- lambda_KD=lambda_MSE=lambda_R=1, lambda_EWC=100
- Warmup epochs = 3
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.haes import HAES
from models.constraints import EWCLoss
from data.dataset import collate_variable_length
from utils.metrics import compute_video_level_ap, compute_video_level_auc, compute_bwt
from utils.logging import save_checkpoint, log_metrics


def _get_primary_metric(metrics, config):
    """Select dataset-appropriate metric for BWT performance matrix.
    XD-Violence uses AP, UCF-Crime uses AUC (Section IV-A)."""
    primary = config.get("primary_metric", "AP")
    if primary == "AUC":
        return metrics.get("mean_auc", 0)
    return metrics.get("mean_ap", 0)


class IncrementalTrainer:
    """
    Incremental trainer for HAES.

    Manages the training across multiple incremental phases:
    - XD-Violence: 6 phases
    - UCF-Crime: 13 phases
    - No exemplar replay from previous phases (strict privacy)
    """

    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device

        # Reproducibility (Section IV-B: seed=42)
        seed = config.get("seed", 42)
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Training hyperparameters
        train_cfg = config.get("training", {})
        self.epochs_per_phase = train_cfg.get("epochs_per_phase", 30)
        self.batch_size = train_cfg.get("batch_size", 64)
        self.learning_rate = train_cfg.get("learning_rate", 5e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.num_workers = train_cfg.get("num_workers", 4)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 3)

        # Output directories
        self.log_dir = config.get("log_dir", "./logs")
        self.checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Model
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # Tracking
        self.current_phase = 0
        self.phase_performances = []  # [T x T] for BWT computation
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # EWC Fisher estimation
        self.ewc_num_batches = train_cfg.get("ewc_fisher_batches", 100)

    def initialize_model(self, num_classes):
        """Initialize HAES model for the given number of classes."""
        model_config = {
            "input_dim": self.config.get("feature_dim", 512),
            "latent_dim": self.config.get("latent_dim", 512),
            "num_classes": num_classes,
            "num_families": self.config.get("num_families", 4),
            "experts_per_family": self.config.get("experts_per_family", 3),
            "top_k1": self.config.get("routing_top_k1", 2),
            "top_k2": self.config.get("routing_top_k2", 2),
            "expert_num_heads": self.config.get("expert_num_heads", 8),
            "expert_ffn_dim": self.config.get("expert_ffn_dim", 2048),
            "expert_num_layers": self.config.get("expert_num_layers", 2),
            "dropout": self.config.get("dropout", 0.1),
            "max_seq_len": self.config.get("max_seq_len", 200),
            "top_k_segments": self.config.get("top_k_segments", 3),
            "temperature": self.config.get("temperature", 4.0),
            "top_k_class": self.config.get("top_k_class", 5),
            "lambda_kd": self.config.get("lambda_kd", 1.0),
            "lambda_mse": self.config.get("lambda_mse", 1.0),
            "lambda_r": self.config.get("lambda_r", 1.0),
            "lambda_ewc": self.config.get("lambda_ewc", 100.0),
            "lambda_temp": self.config.get("lambda_temp", 0.1),
            "ewc_fisher_decay": self.config.get("ewc_fisher_decay", 0.9),
            "warmup_epochs": self.warmup_epochs,
            "elm_config": self.config.get("elm", {}),
        }

        self.model = HAES(model_config)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs_per_phase,
            eta_min=1e-6,
        )

    def train_phase(self, phase_idx, train_loader, test_loader,
                    seen_categories, phase_categories, logger):
        """
        Train one incremental phase.

        Args:
            phase_idx: phase index (0-based)
            train_loader: DataLoader for current phase training data
            test_loader: DataLoader for cumulative test data (all seen categories)
            seen_categories: list of all category names seen so far
            phase_categories: list of new category names introduced in this phase
            logger: logging.Logger instance

        Returns:
            phase_metrics: dict of evaluation metrics for this phase
        """
        self.current_phase = phase_idx
        num_classes = len(seen_categories)

        if phase_idx == 0:
            # Phase 1: Initialize model from scratch
            self.initialize_model(num_classes)
            logger.info(f"Phase {phase_idx + 1}: Initialized HAES with {num_classes} classes")
            logger.info(f"Categories: {phase_categories}")
        else:
            # Update model for new classes
            self._expand_model_for_new_classes(num_classes)
            logger.info(f"Phase {phase_idx + 1}: Expanded to {num_classes} classes")
            logger.info(f"New categories: {phase_categories}")
            logger.info(f"All seen categories: {seen_categories}")

        # Set model to training mode
        self.model.train()

        # Initialize ELM for this phase
        self.model.elm.start_phase(phase_idx)

        # Warmup applies to first incremental phase (phase 1+), not phase 0
        # Phase 0 has no teacher, so warmup has no effect there
        # Phase 1+ has a teacher; warmup prevents premature noise solidification
        # during the first warmup_epochs when new categories are introduced
        self.model.in_warmup = (phase_idx > 0)  # Activate warmup for incremental phases

        best_loss = float("inf")
        phase_losses = []

        for epoch in range(self.epochs_per_phase):
            # Disable warmup after warmup_epochs
            if self.model.in_warmup and epoch >= self.warmup_epochs:
                self.model.in_warmup = False
                logger.info(f"Phase {phase_idx + 1} Epoch {epoch + 1}: Warmup ended, activating all constraints")

            epoch_loss, epoch_loss_dict = self._train_epoch(
                train_loader, epoch, phase_idx, logger
            )
            phase_losses.append(epoch_loss)

            # Learning rate scheduling
            self.scheduler.step()

            # ELM structural adaptation at end of epoch
            elm_action = self.model.elm.step_epoch(epoch_loss)
            if elm_action:
                logger.info(f"Phase {phase_idx + 1} Epoch {epoch + 1}: ELM - {elm_action}")

            # Validation & checkpointing
            if epoch % 5 == 0 or epoch == self.epochs_per_phase - 1:
                val_metrics = self.validate(test_loader, seen_categories, logger)
                val_metrics["epoch"] = epoch
                val_metrics["phase"] = phase_idx
                val_metrics["train_loss"] = epoch_loss

                logger.info(
                    f"Phase {phase_idx + 1} Epoch {epoch + 1}/{self.epochs_per_phase}: "
                    f"Loss={epoch_loss:.4f}, "
                    f"{self._format_metrics(val_metrics)}"
                )

                # TensorBoard logging
                self._log_to_tensorboard(phase_idx, epoch, epoch_loss_dict, val_metrics)

                # Save best checkpoint
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_checkpoint(
                        self.model, self.optimizer,
                        epoch, phase_idx, val_metrics,
                        self.checkpoint_dir,
                        f"haes_phase{phase_idx}_best.pt"
                    )

        # End of phase: Update EWC anchor (snapshot current parameters as old reference)
        self.model.update_ewc_anchor()

        # Estimate Fisher information on current phase data
        logger.info(f"Phase {phase_idx + 1}: Estimating Fisher information...")
        new_fisher = self.model.ewc_loss.estimate_fisher(
            train_loader,
            num_batches=self.ewc_num_batches
        )
        self.model.ewc_loss.update_fisher(new_fisher)

        # Set teacher for next phase
        if phase_idx < self.config.get("total_phases", 6) - 1:
            self.model.set_teacher({
                k: v.clone() for k, v in self.model.state_dict().items()
                if k.startswith("hmoe") or k.startswith("scoring_head")
            })

        # Final evaluation on this phase
        final_metrics = self.validate(test_loader, seen_categories, logger)
        logger.info(
            f"Phase {phase_idx + 1} Final: {self._format_metrics(final_metrics)}"
        )
        logger.info(f"ELM Stats: {self.model.elm.get_stats()}")

        return final_metrics

    def _train_epoch(self, train_loader, epoch, phase_idx, logger):
        """Train for one epoch."""
        total_loss = 0.0
        total_loss_dict = {}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Phase {phase_idx + 1} Epoch {epoch + 1}")

        for batch_idx, (features, labels, lengths) in enumerate(pbar):
            features = features.to(self.device)  # [B, T_seg, 512]
            labels = labels.to(self.device)  # [B]

            # Forward pass through Student
            outputs = self.model(features)

            # Get Teacher outputs (if available)
            teacher_outputs = None
            if self.model.teacher is not None:
                teacher_outputs = self.model.get_teacher_outputs(features)

            # Compute total loss (Eq. 22)
            total_loss_batch, loss_dict = self.model.compute_loss(
                outputs, labels, teacher_outputs, epoch
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Record ELM batch statistics
            self.model.elm.record_batch(
                outputs["routing_info"],
                features
            )

            # Accumulate metrics
            num_batches += 1
            total_loss += total_loss_batch.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v

            # Update progress bar
            pbar.set_postfix({"loss": f"{total_loss_batch.item():.4f}"})

        # Average over batches
        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_dict = {k: v / max(num_batches, 1)
                         for k, v in total_loss_dict.items()}

        return avg_loss, avg_loss_dict

    @torch.no_grad()
    def validate(self, test_loader, seen_categories, logger=None):
        """
        Validate on test set containing all seen categories.

        Returns:
            metrics: dict with AP (XD-Violence) or AUC (UCF-Crime) metrics
        """
        self.model.eval()

        all_scores = []
        all_labels = []
        all_video_ids = []

        for features, labels, lengths in test_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(features)
            video_scores = outputs["video_scores"]  # [B, C]

            all_scores.append(video_scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        self.model.train()

        # Concatenate all batches
        all_scores = np.concatenate(all_scores, axis=0)  # [N, C]
        all_labels = np.concatenate(all_labels, axis=0)  # [N]

        num_classes = len(seen_categories)

        # Compute per-class and mean metrics
        class_aps, mean_ap = compute_video_level_ap(
            all_scores, all_labels, num_classes
        )
        class_aucs, mean_auc = compute_video_level_auc(
            all_scores, all_labels, num_classes
        )

        metrics = {
            "mean_ap": mean_ap,
            "mean_auc": mean_auc,
            "class_aps": class_aps,
            "class_aucs": class_aucs,
        }

        return metrics

    def _expand_model_for_new_classes(self, new_num_classes):
        """Expand the classification head to accommodate new categories."""
        if self.model is None:
            self.initialize_model(new_num_classes)
            return

        old_head = self.model.scoring_head
        old_num_classes = old_head.num_classes

        if new_num_classes <= old_num_classes:
            return

        # Create new scoring head with expanded output dimension
        new_head = type(old_head)(
            latent_dim=old_head.latent_dim,
            num_classes=new_num_classes,
            top_k=old_head.top_k,
        ).to(self.device)

        # Copy old weights for existing classes
        with torch.no_grad():
            new_head.fc2.weight[:old_num_classes] = old_head.fc2.weight
            new_head.fc2.bias[:old_num_classes] = old_head.fc2.bias
            # New class weights initialized randomly (small values)
            nn.init.xavier_uniform_(new_head.fc2.weight[old_num_classes:])
            nn.init.zeros_(new_head.fc2.bias[old_num_classes:])

        self.model.scoring_head = new_head

        # Recreate optimizer preserving old parameter states where possible
        old_param_ids = {id(p) for p in self.optimizer.param_groups[0]["params"]}
        new_params = list(self.model.parameters())
        new_param_ids = [id(p) for p in new_params]

        # Preserve optimizer state for parameters that haven't changed
        old_state = self.optimizer.state_dict()
        self.optimizer = torch.optim.Adam(
            new_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Restore state for parameters that still exist
        new_state = self.optimizer.state_dict()
        for old_p_id, state in old_state["state"].items():
            for new_p, new_p_id in zip(new_params, new_param_ids):
                if id(new_p) == old_p_id:
                    new_state["state"][new_p_id] = state
                    break
        self.optimizer.load_state_dict(new_state)

        # Update scheduler to point to new optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs_per_phase,
            eta_min=1e-6,
        )

    def _format_metrics(self, metrics):
        """Format metrics dict for logging."""
        parts = []
        if "mean_ap" in metrics:
            parts.append(f"AP={metrics['mean_ap']:.4f}")
        if "mean_auc" in metrics:
            parts.append(f"AUC={metrics['mean_auc']:.4f}")
        return " ".join(parts)

    def _log_to_tensorboard(self, phase, epoch, loss_dict, val_metrics):
        """Log metrics to TensorBoard."""
        step = phase * self.epochs_per_phase + epoch
        for k, v in loss_dict.items():
            self.writer.add_scalar(f"loss/{k}", v, step)
        if "mean_ap" in val_metrics:
            self.writer.add_scalar("val/mean_ap", val_metrics["mean_ap"], step)
        if "mean_auc" in val_metrics:
            self.writer.add_scalar("val/mean_auc", val_metrics["mean_auc"], step)

    def train_all_phases(self, phase_loaders, phase_configs, logger):
        """
        Train across all incremental phases.

        Args:
            phase_loaders: list of (train_loader, test_loader) per phase
            phase_configs: list of (seen_categories, phase_categories) per phase
            logger: logging.Logger instance

        Returns:
            all_phase_results: list of metrics per phase
            bwt: Backward Transfer value
        """
        num_phases = len(phase_loaders)
        all_phase_results = []

        # Performance tracking matrix for BWT
        # performance_matrix[t][i] = AP/AUC on phase i after completing phase t
        self.performance_matrix = np.zeros((num_phases, num_phases))

        for phase_idx in range(num_phases):
            train_loader, test_loader = phase_loaders[phase_idx]
            seen_cats, phase_cats = phase_configs[phase_idx]

            logger.info(f"{'='*60}")
            logger.info(f"Starting Phase {phase_idx + 1}/{num_phases}")
            logger.info(f"New categories: {phase_cats}")
            logger.info(f"All seen: {seen_cats}")
            logger.info(f"Training samples: {len(train_loader.dataset)}")
            logger.info(f"{'='*60}")

            start_time = time.time()

            # Train one phase
            phase_metrics = self.train_phase(
                phase_idx, train_loader, test_loader,
                seen_cats, phase_cats, logger
            )

            phase_time = time.time() - start_time
            logger.info(f"Phase {phase_idx + 1} completed in {phase_time:.1f}s")

            # Record performance after this phase
            # Use dataset-appropriate metric: AP for XD-Violence, AUC for UCF-Crime
            primary_metric = _get_primary_metric(phase_metrics, self.config)
            self.performance_matrix[phase_idx, phase_idx] = primary_metric

            # Evaluate on all previous phases
            for prev_phase in range(phase_idx):
                _, prev_test_loader = phase_loaders[prev_phase]
                prev_seen, _ = phase_configs[prev_phase]
                prev_metrics = self.validate(prev_test_loader, prev_seen, logger)
                prev_primary = _get_primary_metric(prev_metrics, self.config)
                self.performance_matrix[phase_idx, prev_phase] = prev_primary

            all_phase_results.append(phase_metrics)

        # Compute BWT after all phases (Eq. 27)
        bwt = compute_bwt(self.performance_matrix)
        logger.info(f"Final BWT: {bwt:.4f}")

        return all_phase_results, bwt
