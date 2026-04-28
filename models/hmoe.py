"""
Hierarchical Mixture of Experts (HMoE) - Core Module.
Section III-B: Two-level sparse routing with Transformer experts.

Architecture overview:
1. Feature encoding + positional embedding (Eq. 2-4)
2. Two-stage hierarchical gating (Eq. 5-6)
3. Per-expert Transformer processing (Eq. 7)
4. Intra-family fusion (Eq. 8)
5. Cross-family fusion (Eq. 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import HierarchicalGate
from .experts import TransformerExpert


class HMoE(nn.Module):
    """
    Hierarchical Mixture of Experts with two-level sparse routing.

    M expert families, each with N_m Transformer experts.
    Total experts: sum_{m=1}^M N_m.
    Active experts per forward pass: k1 * k2 (e.g., 1 * 2 = 2).
    """

    def __init__(self, input_dim=512, latent_dim=512, num_families=4,
                 experts_per_family=3, top_k1=2, top_k2=2,
                 expert_num_heads=8, expert_ffn_dim=2048,
                 expert_num_layers=2, dropout=0.1, max_seq_len=200):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_families = num_families
        self.experts_per_family = experts_per_family
        self.top_k1 = top_k1
        self.top_k2 = top_k2

        # Store expert architecture for dynamic add/merge operations
        self.expert_num_heads = expert_num_heads
        self.expert_ffn_dim = expert_ffn_dim
        self.expert_num_layers = expert_num_layers
        self.expert_dropout = dropout

        # Feature encoding projection (Eq. 2)
        self.W_enc = nn.Linear(input_dim, latent_dim)
        self.enc_norm = nn.LayerNorm(input_dim)
        self.enc_dropout = nn.Dropout(dropout)

        # Learnable positional embedding (Eq. 3-4)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, latent_dim) * 0.02)

        # Hierarchical gating network
        self.gate = HierarchicalGate(
            dim=latent_dim,
            num_families=num_families,
            experts_per_family=experts_per_family,
            top_k1=top_k1,
            top_k2=top_k2,
            dropout=dropout,
        )

        # Expert pool: M families x N_m experts
        self.experts = nn.ModuleList([
            nn.ModuleList([
                TransformerExpert(
                    dim=latent_dim,
                    num_heads=expert_num_heads,
                    ffn_dim=expert_ffn_dim,
                    num_layers=expert_num_layers,
                    dropout=dropout,
                )
                for _ in range(experts_per_family)
            ])
            for _ in range(num_families)
        ])

        self.dropout = nn.Dropout(dropout)

    def encode_features(self, F_seq):
        """
        Feature encoding with positional embedding (Eq. 2-4).

        Args:
            F_seq: clip features from R3D-18 [B, T_seg, 512]

        Returns:
            X: encoded features with position info [B, T_seg, D]
        """
        B, T_seg, _ = F_seq.shape

        # Normalize and project
        F_norm = self.enc_norm(F_seq)
        z = self.W_enc(F_norm)  # [B, T_seg, D]

        # Add positional embedding
        pos = self.pos_embed[:, :T_seg, :]
        X = self.enc_dropout(F.gelu(z)) + pos

        return X

    def forward(self, F_seq, return_routing=False):
        """
        Forward pass through HMoE.

        Args:
            F_seq: clip features [B, T_seg, 512]
            return_routing: if True, return routing distributions for distillation

        Returns:
            H_fused: fused expert outputs [B, T_seg, D]
            routing_info: (optional) gating distributions
        """
        B, T_seg, _ = F_seq.shape

        # Step 1: Feature encoding
        X = self.encode_features(F_seq)  # [B, T_seg, D]

        # Step 2: Hierarchical routing
        routing_info = self.gate(X)

        # Step 3: Expert computation and fusion
        H_fused = torch.zeros(B, T_seg, self.latent_dim,
                              device=F_seq.device, dtype=F_seq.dtype)

        for b in range(B):
            H_b = torch.zeros(T_seg, self.latent_dim, device=F_seq.device, dtype=F_seq.dtype)
            family_weights = routing_info["g1_selected_weights"][b]  # [k1]

            for m_local, (m_idx_tensor, m_weight) in enumerate(zip(
                routing_info["g1_selected_idx"][b],
                family_weights
            )):
                m = m_idx_tensor.item()
                expert_indices, expert_weights = routing_info["g2_selected"][b][m_local]

                # Intra-family fusion (Eq. 8): sum over activated experts in family m
                H_m = torch.zeros(T_seg, self.latent_dim, device=F_seq.device, dtype=F_seq.dtype)
                for n_local, (n_idx_tensor, n_weight) in enumerate(zip(
                    expert_indices.squeeze(0),
                    expert_weights.squeeze(0)
                )):
                    n = n_idx_tensor.item()
                    H_mn = self.experts[m][n](X[b:b+1])  # [1, T_seg, D]
                    H_m += n_weight * H_mn.squeeze(0)

                # Cross-family fusion (Eq. 9): weighted sum over families
                H_b += m_weight * H_m

            H_fused[b] = H_b

        if return_routing:
            return H_fused, routing_info
        return H_fused

    def get_expert_centroids(self, X):
        """Get centroid feature vectors for all experts (for ELM merging)."""
        centroids = {}
        for m in range(self.num_families):
            for n in range(len(self.experts[m])):
                centroids[(m, n)] = self.experts[m][n].get_centroid(X)
        return centroids

    def add_expert_to_family(self, m):
        """
        Add a new expert to family m (ELM Addition).
        Parameters initialized as mean of sibling experts (Eq. 23).
        """
        if len(self.experts[m]) >= 8:  # Safety cap
            return False

        # Compute mean of sibling expert parameters (Eq. 23)
        new_expert = TransformerExpert(
            dim=self.latent_dim,
            num_heads=self.expert_num_heads,
            ffn_dim=self.expert_ffn_dim,
            num_layers=self.expert_num_layers,
            dropout=self.expert_dropout,
        )

        # Initialize as mean of existing experts (Eq. 23)
        with torch.no_grad():
            existing_params_list = list(self.experts[m][0].parameters())
            for p_idx, new_param in enumerate(new_expert.parameters()):
                stacked = torch.stack([
                    list(e.parameters())[p_idx].data
                    for e in self.experts[m]
                ])
                new_param.copy_(stacked.mean(dim=0))

        self.experts[m].append(new_expert)
        # Update expert gate dimension
        old_gate = self.gate.expert_gates[m]
        new_gate = nn.Linear(self.latent_dim, len(self.experts[m]))
        # Copy old weights
        with torch.no_grad():
            new_gate.weight[:old_gate.experts_per_family] = old_gate.W_g2.weight
            new_gate.bias[:old_gate.experts_per_family] = old_gate.W_g2.bias
        new_gate = new_gate.to(next(self.parameters()).device)
        # Keep as linear layer - wrap in ExpertGate logic
        old_gate.experts_per_family = len(self.experts[m])
        old_gate.W_g2 = new_gate

        return True

    def merge_experts(self, m, i, j, alpha_i, alpha_j):
        """
        Merge two experts in the same family (ELM Merging, Eq. 24).
        theta_merged = (alpha_i * theta_i + alpha_j * theta_j) / (alpha_i + alpha_j)
        """
        # Ensure i < j so removal index doesn't shift
        if i > j:
            i, j = j, i
            alpha_i, alpha_j = alpha_j, alpha_i

        with torch.no_grad():
            for param_i, param_j in zip(
                self.experts[m][i].parameters(),
                self.experts[m][j].parameters()
            ):
                merged = (alpha_i * param_i.data + alpha_j * param_j.data) / (alpha_i + alpha_j + 1e-8)
                param_i.data.copy_(merged)

        # Remove expert j (updates gating dimensions)
        success = self._remove_expert(m, j)
        return success

    def recycle_expert(self, m, n):
        """
        Reset an expert to random initialization (ELM Recycling).
        theta -> N(0, sigma_init^2)
        """
        with torch.no_grad():
            for param in self.experts[m][n].parameters():
                param.normal_(mean=0.0, std=0.02)

    def _remove_expert(self, m, n):
        """Remove expert n from family m. Updates gating dimensions to match."""
        if len(self.experts[m]) <= 2:  # Minimum experts per family
            return False

        # Remove the expert from the ModuleList
        self.experts[m].pop(n)
        new_count = len(self.experts[m])

        # Update Stage-2 expert gate to match the new expert count (Eq. 6)
        old_gate = self.gate.expert_gates[m]
        device = old_gate.W_g2.weight.device
        dtype = old_gate.W_g2.weight.dtype

        new_W_g2 = nn.Linear(self.latent_dim, new_count).to(device=device, dtype=dtype)
        # Preserve weights for remaining experts (shift indices after removed expert)
        with torch.no_grad():
            src_idx = 0
            for dst_idx in range(new_count):
                if src_idx == n:
                    src_idx += 1  # Skip removed expert
                new_W_g2.weight[dst_idx] = old_gate.W_g2.weight[src_idx]
                new_W_g2.bias[dst_idx] = old_gate.W_g2.bias[src_idx]
                src_idx += 1

        old_gate.W_g2 = new_W_g2
        old_gate.experts_per_family = new_count
        return True

    def get_active_expert_count(self):
        """Return total number of active experts across all families."""
        return sum(len(family) for family in self.experts)

    def get_family_sizes(self):
        """Return list of expert counts per family."""
        return [len(family) for family in self.experts]
