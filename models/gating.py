"""
Two-level Hierarchical Gating Networks for HMoE.
Section III-B: Family-level and Expert-level routing with Top-k sparse selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FamilyGate(nn.Module):
    """
    Stage-1: Family-level gating.
    Routes clip features to expert families using Top-k1 sparse selection.

    g^(1) = Softmax(x_bar W_g1 + b_g1) in R^M
    M = TopK(g^(1), k1) -> activated families
    """

    def __init__(self, dim, num_families, top_k=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_families = num_families
        self.top_k = top_k

        self.W_g1 = nn.Linear(dim, num_families)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_g1.weight)
        nn.init.zeros_(self.W_g1.bias)

    def forward(self, x_bar):
        """
        Args:
            x_bar: mean-pooled sequence feature [B, D]

        Returns:
            selected_families: list of activated family indices per sample
            g_normalized: renormalized gating weights [B, k1]
            g_raw: raw softmax distribution [B, M]
        """
        # Softmax gating
        g_raw = F.softmax(self.W_g1(self.dropout(x_bar)), dim=-1)  # [B, M]

        # Top-k selection
        g_topk_vals, g_topk_idx = torch.topk(g_raw, self.top_k, dim=-1)  # [B, k1]

        # Renormalize
        g_normalized = g_topk_vals / (g_topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        return g_topk_idx, g_normalized, g_raw


class ExpertGate(nn.Module):
    """
    Stage-2: Intra-family expert-level gating.
    Within each activated family, selects Top-k2 experts.

    g_m^(2) = Softmax(x_bar W_g2,m + b_g2,m) in R^{N_m}
    E_m = TopK(g_m^(2), k2) -> activated experts within family m
    """

    def __init__(self, dim, experts_per_family, top_k=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        # Register as buffer so it survives state_dict serialization
        self.register_buffer(
            "experts_per_family_tensor",
            torch.tensor(experts_per_family, dtype=torch.long)
        )

        self.W_g2 = nn.Linear(dim, experts_per_family)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    @property
    def experts_per_family(self):
        return self.experts_per_family_tensor.item()

    @experts_per_family.setter
    def experts_per_family(self, value):
        self.experts_per_family_tensor.fill_(value)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_g2.weight)
        nn.init.zeros_(self.W_g2.bias)

    def forward(self, x_bar):
        """
        Args:
            x_bar: mean-pooled sequence feature [B, D]

        Returns:
            selected_experts: activated expert indices [B, k2]
            g_normalized: renormalized weights [B, k2]
            g_raw: raw distribution [B, N_m]
        """
        g_raw = F.softmax(self.W_g2(self.dropout(x_bar)), dim=-1)  # [B, N_m]

        k = min(self.top_k, self.experts_per_family)
        g_topk_vals, g_topk_idx = torch.topk(g_raw, k, dim=-1)

        g_normalized = g_topk_vals / (g_topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        return g_topk_idx, g_normalized, g_raw


class HierarchicalGate(nn.Module):
    """
    Full two-level hierarchical gating network.

    Combined forward pass:
    1. Family-level: select Top-k1 families
    2. Expert-level: within each selected family, select Top-k2 experts
    3. Output gating distributions for routing and routing-distillation loss
    """

    def __init__(self, dim, num_families, experts_per_family,
                 top_k1=2, top_k2=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_families = num_families
        self.experts_per_family = experts_per_family
        self.top_k1 = top_k1
        self.top_k2 = top_k2

        # Stage-1 gate
        self.family_gate = FamilyGate(dim, num_families, top_k1, dropout)

        # Stage-2 gates: one per family
        self.expert_gates = nn.ModuleList([
            ExpertGate(dim, experts_per_family, top_k2, dropout)
            for _ in range(num_families)
        ])

    def forward(self, X):
        """
        Args:
            X: sequence features [B, T_seg, D]

        Returns:
            routing_info: dict with all routing information
        """
        B, T_seg, D = X.shape

        # Sequence aggregation (Eq. 5): mean over time
        x_bar = X.mean(dim=1)  # [B, D]

        # Stage 1: Family-level routing
        family_idx, family_weights, g1_raw = self.family_gate(x_bar)

        # Stage 2: Within-family expert routing
        routing_info = {
            "x_bar": x_bar,
            "g1_raw": g1_raw,  # [B, M] - for routing distillation
            "g1_selected_idx": family_idx,  # [B, k1]
            "g1_selected_weights": family_weights,  # [B, k1]
            "g2_raw": [],  # per-family raw distributions
            "g2_selected": [],  # per-family (indices, weights)
        }

        for b in range(B):
            batch_g2_raw = []
            batch_g2_selected = []
            for m_idx in family_idx[b]:
                m = m_idx.item()
                expert_idx, expert_weights, g2_raw = self.expert_gates[m](x_bar[b:b+1])
                batch_g2_raw.append(g2_raw)
                batch_g2_selected.append((expert_idx, expert_weights))
            routing_info["g2_raw"].append(batch_g2_raw)
            routing_info["g2_selected"].append(batch_g2_selected)

        return routing_info

    def get_load_balance_loss(self, g1_raw):
        """Auxiliary load-balance loss: sum_m (mean_m g1)^2 to discourage collapse."""
        mean_g = g1_raw.mean(dim=0)  # [M]
        return (mean_g ** 2).sum()
