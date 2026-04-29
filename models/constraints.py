"""
Incremental Learning Constraints and Distillation for HAES.
Section III-C: Four-level defense against catastrophic forgetting.

Constraints:
1. L_KD: Output distribution distillation (Eq. 10-12)
2. L_MSE: Feature consistency (Eq. 13)
3. L_R-KL: Routing distribution preservation (Eq. 14-17)
4. L_EWC: Parameter importance regularization (Eq. 18-19)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Output distribution distillation with truncated normalized KL.
    Section III-C Eq. 10-12.

    Uses temperature-scaled softmax and Top-k class support set
    to focus on high-confidence classes.
    """

    def __init__(self, temperature=4.0, top_k_class=5):
        super().__init__()
        self.temperature = temperature
        self.top_k_class = top_k_class

    def forward(self, student_logits, teacher_logits, reduction="mean"):
        """
        Args:
            student_logits: [B, T_seg, C]
            teacher_logits: [B, T_seg, C]
            reduction: "mean" (scalar) or "none" (per-sample [B])

        Returns:
            L_KD: scalar loss (reduction="mean") or [B] (reduction="none")
        """
        B, T_seg, C = student_logits.shape
        C_t = teacher_logits.shape[-1]

        # Align class dimensions: student may have more classes than teacher
        # after incremental expansion. Slice student to teacher's class set
        # so softmax is over identical support (valid KL divergence).
        if C > C_t:
            student_logits = student_logits[:, :, :C_t]

        # Temperature-scaled softmax (Eq. 10)
        p_s = F.softmax(student_logits / self.temperature, dim=-1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Truncated normalized KL (Eq. 11-12)
        # k must not exceed either student or teacher class dimension
        C_t = p_t.shape[-1]
        k = min(self.top_k_class, C, C_t)
        _, topk_idx = torch.topk(p_t, k, dim=-1)  # [B, T_seg, k]

        per_sample_kl = torch.zeros(B, device=student_logits.device)
        for t in range(T_seg):
            p_s_t = p_s[:, t, :]  # [B, C]
            p_t_t = p_t[:, t, :]  # [B, C]
            topk_t = topk_idx[:, t, :]  # [B, k]

            # Gather and renormalize on support set (Eq. 11)
            p_s_trunc = torch.gather(p_s_t, 1, topk_t)
            p_t_trunc = torch.gather(p_t_t, 1, topk_t)
            p_s_trunc = p_s_trunc / (p_s_trunc.sum(dim=-1, keepdim=True) + 1e-8)
            p_t_trunc = p_t_trunc / (p_t_trunc.sum(dim=-1, keepdim=True) + 1e-8)

            # KL divergence per sample
            kl = (p_t_trunc * (p_t_trunc + 1e-8).log() -
                  p_t_trunc * (p_s_trunc + 1e-8).log()).sum(dim=-1)  # [B]
            per_sample_kl += kl

        # Average over time
        per_sample_kl = per_sample_kl / T_seg

        # Temperature scaling (Eq. 12)
        per_sample_kd = (self.temperature ** 2) * per_sample_kl  # [B]

        if reduction == "none":
            return per_sample_kd
        return per_sample_kd.mean()


class FeatureConsistencyLoss(nn.Module):
    """
    Feature-level consistency via MSE between normalized fused representations.
    Section III-C Eq. 13.

    L_MSE = (1 / (T_seg * D)) * ||H_fused^S - H_fused^T||_F^2
    """

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: [B, T_seg, D]
            teacher_features: [B, T_seg, D]

        Returns:
            L_MSE: scalar loss
        """
        # Eq. 13: L_MSE = (1/(T_seg*D)) * ||H_bar^S - H_bar^T||_F^2
        # Per-timestep L2 normalization: H_bar = H / ||H||_2
        s_norm = torch.nn.functional.normalize(student_features, p=2, dim=-1)
        t_norm = torch.nn.functional.normalize(teacher_features, p=2, dim=-1)
        B, T_seg, D = student_features.shape
        loss = ((s_norm - t_norm) ** 2).sum() / (B * T_seg * D)
        return loss


class RoutingDistillationLoss(nn.Module):
    """
    Routing distribution preservation loss.
    Section III-C Eq. 14-17.

    L_R-KL = L_R^(1) + L_R^(2)
    where L_R^(1) is family-level and L_R^(2) is expert-level KL.
    """

    def __init__(self, top_k1=2, top_k2=2):
        super().__init__()
        self.top_k1 = top_k1
        self.top_k2 = top_k2

    def forward(self, routing_s, routing_t):
        """
        Args:
            routing_s: Student routing info dict
            routing_t: Teacher routing info dict

        Returns:
            L_R_KL: scalar routing distillation loss
        """
        # Stage 1: Family-level routing KL (Eq. 15)
        g1_s = routing_s["g1_raw"]  # [B, M]
        g1_t = routing_t["g1_raw"]  # [B, M]

        B, M = g1_s.shape
        k1 = min(self.top_k1, M)

        # Truncated KL on Top-k1 families (Eq. 14)
        _, topk_idx = torch.topk(g1_t, k1, dim=-1)

        g1_s_trunc = torch.gather(g1_s, 1, topk_idx)
        g1_t_trunc = torch.gather(g1_t, 1, topk_idx)
        g1_s_trunc = g1_s_trunc / (g1_s_trunc.sum(dim=-1, keepdim=True) + 1e-8)
        g1_t_trunc = g1_t_trunc / (g1_t_trunc.sum(dim=-1, keepdim=True) + 1e-8)

        l_r1 = (g1_t_trunc * (g1_t_trunc + 1e-8).log() -
                g1_t_trunc * (g1_s_trunc + 1e-8).log()).sum(dim=-1).mean()

        # Stage 2: Expert-level routing KL (Eq. 16)
        # Match by actual family index, not position in top-k ranking,
        # since student and teacher may select different top-k families.
        l_r2 = 0.0
        count = 0
        for b in range(B):
            # Build family-ID -> g2_raw position lookup for student
            s_family_to_pos = {}
            for pos, m_idx_tensor in enumerate(routing_s["g1_selected_idx"][b]):
                s_family_to_pos[m_idx_tensor.item()] = pos

            # Build family-ID -> g2_raw position lookup for teacher
            t_family_to_pos = {}
            for pos, m_idx_tensor in enumerate(routing_t["g1_selected_idx"][b]):
                t_family_to_pos[m_idx_tensor.item()] = pos

            # Compare expert distributions only for families selected by BOTH models
            for m_actual in range(k1):
                m_t = topk_idx[b, m_actual].item()  # teacher's top-k family ID
                if m_t in s_family_to_pos and m_t in t_family_to_pos:
                    s_pos = s_family_to_pos[m_t]
                    t_pos = t_family_to_pos[m_t]
                    g2_s_raw = routing_s["g2_raw"][b][s_pos]  # [1, N_m]
                    g2_t_raw = routing_t["g2_raw"][b][t_pos]  # [1, N_m]

                    N_m_actual = min(g2_s_raw.size(-1), g2_t_raw.size(-1))
                    k2 = min(self.top_k2, N_m_actual)

                    # Truncate both to common dimension before top-k to prevent
                    # index out-of-bounds when teacher has more experts than student
                    # (e.g., after ELM merge/recycle during incremental phase)
                    g2_t_raw = g2_t_raw[:, :N_m_actual]
                    g2_s_raw = g2_s_raw[:, :N_m_actual]

                    _, topk2_idx = torch.topk(g2_t_raw, k2, dim=-1)

                    g2_s_trunc = torch.gather(g2_s_raw, 1, topk2_idx)
                    g2_t_trunc = torch.gather(g2_t_raw, 1, topk2_idx)
                    g2_s_trunc = g2_s_trunc / (g2_s_trunc.sum(dim=-1, keepdim=True) + 1e-8)
                    g2_t_trunc = g2_t_trunc / (g2_t_trunc.sum(dim=-1, keepdim=True) + 1e-8)

                    l_r2 += (g2_t_trunc * (g2_t_trunc + 1e-8).log() -
                             g2_t_trunc * (g2_s_trunc + 1e-8).log()).sum(dim=-1).mean()
                    count += 1

        if count > 0:
            l_r2 /= count

        return l_r1 + l_r2


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation (EWC) regularization.
    Section III-C Eq. 18-19.

    L_EWC = sum_i F_i (theta_i - theta_i^*)^2

    Uses diagonal Fisher information accumulated from previous phases.
    Implements online-EWC with decay factor for multi-phase accumulation.
    """

    def __init__(self, model, fisher_decay=0.9):
        super().__init__()
        self.model = model
        self.fisher_decay = fisher_decay
        self.fisher_diag = {}  # param_name -> Fisher diagonal
        self.anchor_params = {}  # param_name -> anchor value (theta^*)

    def estimate_fisher(self, dataloader, num_batches=100):
        """
        Estimate diagonal Fisher information matrix on old task data.
        Eq. 18: F_i = E[(d/dtheta_i log p(y|x; theta))^2]
        """
        device = next(self.model.parameters()).device
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        self.model.eval()
        for batch_idx, (features, labels, lengths) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            features = features.to(device)
            labels = labels.to(device)

            self.model.zero_grad()
            outputs = self.model(features)
            log_likelihood = F.log_softmax(outputs["video_logits"], dim=-1)
            log_likelihood = log_likelihood.gather(1, labels.unsqueeze(1)).mean()

            # Compute gradients of log-likelihood
            grads = torch.autograd.grad(log_likelihood, self.model.parameters(),
                                        retain_graph=False)
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                if name in fisher and grad is not None:
                    fisher[name] += grad.detach() ** 2

        # Normalize by actual number of batches processed
        actual_batches = min(len(dataloader), num_batches)
        if actual_batches > 0:
            for name in fisher:
                fisher[name] /= actual_batches

        self.model.train()
        return fisher

    def update_fisher(self, new_fisher):
        """
        Update Fisher with online-EWC decay (paper III-C: "Fisher matrix is
        recomputed once per phase and added to the previous-phase Fisher
        with a decay factor").
        F_new = decay * F_old + F_phase
        """
        if not self.fisher_diag:
            self.fisher_diag = new_fisher
        else:
            for name in self.fisher_diag:
                if name in new_fisher:
                    self.fisher_diag[name] = (
                        self.fisher_decay * self.fisher_diag[name] +
                        new_fisher[name]
                    )

    def set_anchor(self):
        """Set current model parameters as the EWC anchor."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.anchor_params[name] = param.data.clone()

    def forward(self):
        """
        Compute EWC regularization loss (Eq. 19).

        Handles shape expansion after classification head grows:
        only the overlapping parameter region receives the EWC penalty.

        Returns:
            L_EWC: scalar loss
        """
        if not self.fisher_diag or not self.anchor_params:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = 0.0
        for name, param in self.model.named_parameters():
            if name not in self.fisher_diag or name not in self.anchor_params:
                continue
            fisher_val = self.fisher_diag[name]
            anchor_val = self.anchor_params[name]
            # Handle shape expansion after classification head grows
            if param.shape != anchor_val.shape:
                # Slice both to the overlapping dimensions
                slices = tuple(
                    slice(0, min(ps, av))
                    for ps, av in zip(param.shape, anchor_val.shape)
                )
                loss += (fisher_val[slices] *
                         (param[slices] - anchor_val[slices]) ** 2).sum()
            else:
                loss += (fisher_val * (param - anchor_val) ** 2).sum()

        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency constraint.
    Reduces video-level localization noise by enforcing
    KL divergence between prediction distributions of adjacent time windows.
    """

    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size

    def forward(self, segment_logits):
        """
        Args:
            segment_logits: [B, T_seg, C]

        Returns:
            L_temp: scalar loss
        """
        B, T_seg, C = segment_logits.shape
        if T_seg < 2:
            return torch.tensor(0.0, device=segment_logits.device)

        p = F.softmax(segment_logits, dim=-1)

        loss = 0.0
        for t in range(T_seg - 1):
            p_t = p[:, t, :]
            p_next = p[:, t + 1, :]
            kl = (p_t * ((p_t + 1e-8) / (p_next + 1e-8)).log()).sum(dim=-1)
            loss += kl.mean()

        return loss / (T_seg - 1)


class NoiseSuppressionModule(nn.Module):
    """
    Entropy-weighted noise suppression.
    Section III-C: Distillation weights decay exponentially with prediction entropy.

    Theorem 3: Noise samples exhibit higher entropy, receive lower distillation weight.
    """

    def __init__(self, entropy_threshold=1.5):
        super().__init__()
        self.entropy_threshold = entropy_threshold

    def compute_entropy_weights(self, logits):
        """
        Compute per-sample entropy and corresponding attenuation weights.

        Args:
            logits: [B, T_seg, C]

        Returns:
            weights: [B] attenuation weights for distillation
            entropy: [B] per-sample prediction entropy
        """
        p = F.softmax(logits, dim=-1)
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1).mean(dim=1)  # [B]

        # Exponential decay for high-entropy samples
        weights = torch.exp(-entropy / self.entropy_threshold)

        return weights, entropy
