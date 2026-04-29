"""
Hierarchical Adaptive Expert System (HAES) - Full Model.
Integrates all components from Section III:

Components:
1. Feature Extraction (Section III-A): Frozen R3D-18 backbone
2. HMoE (Section III-B): Two-level sparse routing with Transformer experts
3. Incremental Constraints (Section III-C): KD, MSE, Routing-KL, EWC
4. Anomaly Scoring (Section III-D): MLP head + Top-K MIL aggregation
5. ELM (Section III-E): Expert addition, merging, recycling

Total loss (Eq. 22):
L_total = L_cls + lambda_KD*L_KD + lambda_MSE*L_MSE
         + lambda_R*L_R-KL + lambda_EWC*L_EWC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hmoe import HMoE
from .constraints import (
    DistillationLoss,
    FeatureConsistencyLoss,
    RoutingDistillationLoss,
    EWCLoss,
    TemporalConsistencyLoss,
    NoiseSuppressionModule,
)
from .elm import ExpertLifecycleManager


class AnomalyScoringHead(nn.Module):
    """
    Anomaly Scoring and Localization Module (Section III-D).

    MLP head for clip-level scoring:
    s_t = w_2^T * phi(LN(h_t) * W_1 + b_1) + b_2
    a_t = sigma(s_t)

    Top-K MIL aggregation for video-level scoring (Eq. 20):
    S_vid = (1/K) * sum_{t in TopK({s_t}, K)} s_t
    A_vid = sigma(S_vid)
    """

    def __init__(self, latent_dim, num_classes, top_k=3, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.top_k = top_k

        self.norm = nn.LayerNorm(latent_dim)
        self.fc1 = nn.Linear(latent_dim, latent_dim // 2)
        self.fc2 = nn.Linear(latent_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_fused):
        """
        Args:
            H_fused: fused expert features [B, T_seg, D]

        Returns:
            segment_logits: clip-level logits [B, T_seg, C]
            segment_scores: clip-level anomaly probabilities [B, T_seg, C]
            video_logits: video-level logits [B, C]
            video_scores: video-level anomaly probabilities [B, C]
        """
        B, T_seg, D = H_fused.shape

        # Clip-level scoring
        h_norm = self.norm(H_fused)  # [B, T_seg, D]
        h_hidden = F.gelu(self.fc1(self.dropout(h_norm)))  # [B, T_seg, D//2]
        segment_logits = self.fc2(self.dropout(h_hidden))  # [B, T_seg, C]
        segment_scores = torch.sigmoid(segment_logits)  # [B, T_seg, C]

        # Top-K MIL aggregation (Eq. 20)
        K = min(self.top_k, T_seg)
        # Take top-K across each class dimension
        topk_vals, _ = torch.topk(segment_logits, K, dim=1)  # [B, K, C]
        video_logits = topk_vals.mean(dim=1)  # [B, C]
        video_scores = torch.sigmoid(video_logits)  # [B, C]

        return {
            "segment_logits": segment_logits,
            "segment_scores": segment_scores,
            "video_logits": video_logits,
            "video_scores": video_scores,
        }


class HAES(nn.Module):
    """
    Hierarchical Adaptive Expert System (HAES).

    Full model combining HMoE with incremental learning constraints
    and Expert Lifecycle Management for weakly-supervised
    incremental violence detection.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Model dimensions ---
        input_dim = config.get("input_dim", 512)
        latent_dim = config.get("latent_dim", 512)
        num_classes = config.get("num_classes", 13)
        num_families = config.get("num_families", 4)
        experts_per_family = config.get("experts_per_family", 3)
        top_k1 = config.get("top_k1", 2)
        top_k2 = config.get("top_k2", 2)
        expert_num_heads = config.get("expert_num_heads", 8)
        expert_ffn_dim = config.get("expert_ffn_dim", 2048)
        expert_num_layers = config.get("expert_num_layers", 2)
        dropout = config.get("dropout", 0.1)
        max_seq_len = config.get("max_seq_len", 200)

        # --- HMoE module ---
        self.hmoe = HMoE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_families=num_families,
            experts_per_family=experts_per_family,
            top_k1=top_k1,
            top_k2=top_k2,
            expert_num_heads=expert_num_heads,
            expert_ffn_dim=expert_ffn_dim,
            expert_num_layers=expert_num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # --- Anomaly scoring head ---
        top_k_segments = config.get("top_k_segments", 3)
        self.scoring_head = AnomalyScoringHead(
            latent_dim=latent_dim,
            num_classes=num_classes,
            top_k=top_k_segments,
            dropout=dropout,
        )

        # --- Constraint losses ---
        temperature = config.get("temperature", 4.0)
        top_k_class = config.get("top_k_class", 5)
        self.kd_loss = DistillationLoss(
            temperature=temperature,
            top_k_class=top_k_class,
        )
        self.mse_loss = FeatureConsistencyLoss()
        self.routing_loss = RoutingDistillationLoss(
            top_k1=top_k1,
            top_k2=top_k2,
        )
        self.ewc_loss = EWCLoss(
            model=self,
            fisher_decay=config.get("ewc_fisher_decay", 0.9),
        )
        self.temp_loss = TemporalConsistencyLoss(window_size=3)
        self.noise_module = NoiseSuppressionModule(entropy_threshold=1.5)

        # --- Loss weights ---
        # Eq. 22 has 5 terms: cls + lambda_KD + lambda_MSE + lambda_R + lambda_EWC.
        # `lambda_balance` is an optional auxiliary controlling the load-balance
        # term mentioned in paper line 441 ("a load-balance term Σ_m(g_bar_m^(1))^2");
        # default 0 so the optimised objective matches Eq. 22 exactly.
        self.lambda_kd = config.get("lambda_kd", 1.0)
        self.lambda_mse = config.get("lambda_mse", 1.0)
        self.lambda_r = config.get("lambda_r", 1.0)
        self.lambda_ewc = config.get("lambda_ewc", 100.0)
        self.lambda_temp = config.get("lambda_temp", 0.1)
        self.lambda_balance = config.get("lambda_balance", 0.0)

        # --- ELM manager ---
        elm_config = config.get("elm_config", {})
        self.elm = ExpertLifecycleManager(elm_config, self.hmoe)

        # --- Teacher model (frozen previous phase) ---
        self.teacher = None

        # --- Warm-up flag ---
        self.in_warmup = True
        self.warmup_epochs = config.get("warmup_epochs", 3)

    def forward(self, F_seq, return_all=False):
        """
        Forward pass through HAES.

        Args:
            F_seq: clip feature sequence [B, T_seg, 512]
            return_all: if True, return all intermediate outputs

        Returns:
            outputs: dict with scoring results and optionally routing info
        """
        # HMoE forward pass (Section III-B)
        H_fused, routing_info = self.hmoe(F_seq, return_routing=True)

        # Anomaly scoring (Section III-D)
        scores = self.scoring_head(H_fused)

        # Combine outputs
        outputs = {**scores, "H_fused": H_fused, "routing_info": routing_info}

        return outputs

    def get_teacher_outputs(self, F_seq):
        """Get outputs from frozen Teacher model."""
        if self.teacher is None:
            return None
        with torch.no_grad():
            return self.teacher(F_seq, return_all=True)

    def compute_loss(self, outputs, labels, teacher_outputs=None, epoch=0):
        """
        Compute total training loss (Eq. 22).

        Args:
            outputs: Student model outputs dict
            labels: video-level labels [B]
            teacher_outputs: Teacher model outputs (None for Phase 1)
            epoch: current epoch number

        Returns:
            total_loss: scalar
            loss_dict: breakdown of individual losses
        """
        # Classification loss (Eq. 21):
        #   L_cls = -Σ_c 1[y=c] log A_vid^(c) where A_vid^(c) = σ(S_vid^(c))
        # Per-class independent sigmoid (paper Eq. 20), then -log of true class.
        # This matches the paper's literal σ-then-indicator formulation rather
        # than softmax cross-entropy.
        video_logits = outputs["video_logits"]  # [B, C]
        video_probs = torch.sigmoid(video_logits)  # [B, C]
        true_class_probs = video_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
        loss_cls = -torch.log(true_class_probs + 1e-8).mean()

        total_loss = loss_cls
        loss_dict = {"cls": loss_cls.item()}

        # Knowledge distillation losses (Section III-C)
        # During warmup: uniform weights (all ones) per paper "Uniform distillation
        # weights to avoid premature noise solidification" (Section IV-B)
        # After warmup: entropy-weighted per-sample noise suppression (Theorem 3)
        if teacher_outputs is not None:
            if self.in_warmup:
                entropy_weights = torch.ones(
                    outputs["segment_logits"].size(0),
                    device=outputs["segment_logits"].device
                )
            else:
                entropy_weights, _ = self.noise_module.compute_entropy_weights(
                    outputs["segment_logits"]
                )  # [B]

            # Output KD (Eq. 12) - per-sample entropy weighting (Theorem 3)
            per_sample_kd = self.kd_loss(
                outputs["segment_logits"],
                teacher_outputs["segment_logits"],
                reduction="none",
            )  # [B]
            loss_kd = (per_sample_kd * entropy_weights).mean()

            # Feature consistency (Eq. 13)
            loss_mse = self.mse_loss(
                outputs["H_fused"],
                teacher_outputs["H_fused"]
            )

            # Routing preservation (Eq. 17)
            loss_r = self.routing_loss(
                outputs["routing_info"],
                teacher_outputs["routing_info"]
            )

            # EWC regularization (Eq. 19) — gated by the same noise-suppression
            # entropy mask used for KD (Section III-C: "two losses are gated by
            # a confidence mask that excludes high-entropy clips from both
            # terms"). The mean entropy weight scales the parameter penalty.
            loss_ewc = self.ewc_loss() * entropy_weights.mean()

            # Weighted combination (Eq. 22)
            total_loss += (
                self.lambda_kd * loss_kd +
                self.lambda_mse * loss_mse +
                self.lambda_r * loss_r +
                self.lambda_ewc * loss_ewc
            )

            loss_dict.update({
                "kd": loss_kd.item(),
                "mse": loss_mse.item(),
                "r_kl": loss_r.item(),
                "ewc": loss_ewc.item(),
            })

            # Temporal consistency loss (Section III-C): only in incremental phases
            loss_temp = self.temp_loss(outputs["segment_logits"])
            total_loss += self.lambda_temp * loss_temp
            loss_dict["temp"] = loss_temp.item()

        # Load-balance term Σ_m (g_bar_m^(1))^2 from paper line 441. Eq. 22
        # does NOT enumerate it, so by default lambda_balance = 0 keeps L_total
        # strictly equal to the five paper-listed terms; setting lambda_balance
        # > 0 in config enables the auxiliary term mentioned in HMoE design.
        loss_balance = self.hmoe.gate.get_load_balance_loss(
            outputs["routing_info"]["g1_raw"]
        )
        if self.lambda_balance > 0:
            total_loss = total_loss + self.lambda_balance * loss_balance
        loss_dict["balance"] = loss_balance.item()
        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict

    def set_teacher(self, teacher_state_dict):
        """Set teacher model from a frozen snapshot of the previous phase."""
        # Create a deep copy of config with updated num_classes to match
        # the state_dict shapes (handles expanded classification heads)
        import copy
        teacher_config = copy.deepcopy(self.config)

        # Infer num_classes from state dict to handle expansion
        if "scoring_head.fc2.weight" in teacher_state_dict:
            teacher_config["num_classes"] = teacher_state_dict["scoring_head.fc2.weight"].shape[0]
        if "scoring_head.fc2.bias" in teacher_state_dict:
            teacher_config["num_classes"] = teacher_state_dict["scoring_head.fc2.bias"].shape[0]

        self.teacher = HAES(teacher_config)
        self.teacher.load_state_dict(teacher_state_dict, strict=False)
        # Move teacher to same device as student model
        device = next(self.parameters()).device
        self.teacher.to(device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_ewc_anchor(self):
        """Set current parameters as EWC anchor and update Fisher."""
        self.ewc_loss.set_anchor()

    def get_state_dict(self):
        """Get serializable state dict excluding teacher."""
        return {
            "hmoe": self.hmoe.state_dict(),
            "scoring_head": self.scoring_head.state_dict(),
            "ewc_fisher": {k: v.cpu() for k, v in
                           self.ewc_loss.fisher_diag.items()},
            "ewc_anchor": {k: v.cpu() for k, v in
                           self.ewc_loss.anchor_params.items()},
        }

    def load_state(self, state_dict):
        """Load state including EWC buffers."""
        self.hmoe.load_state_dict(state_dict["hmoe"], strict=False)
        self.scoring_head.load_state_dict(state_dict["scoring_head"], strict=False)
        if "ewc_fisher" in state_dict:
            device = next(self.parameters()).device
            self.ewc_loss.fisher_diag = {
                k: v.to(device) for k, v in state_dict["ewc_fisher"].items()
            }
        if "ewc_anchor" in state_dict:
            device = next(self.parameters()).device
            self.ewc_loss.anchor_params = {
                k: v.to(device) for k, v in state_dict["ewc_anchor"].items()
            }
