"""
Transformer Expert Blocks for HMoE.
Section III-B Eq. 7: Each expert E_{m,n} is a Transformer Encoder Block.
"""

import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for temporal modeling."""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, dim, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerExpertBlock(nn.Module):
    """
    Single Transformer Encoder block used as an expert.
    Section III-B Eq. 7: H_{m,n} = E_{m,n}(X)
    """

    def __init__(self, dim, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerExpert(nn.Module):
    """
    Full Transformer Expert: stack of TransformerEncoderBlocks.

    Each expert models different temporal patterns:
    - Some focus on long-term dependencies
    - Some focus on local abrupt changes
    - Some focus on fine-grained interactions
    """

    def __init__(self, dim, num_heads=8, ffn_dim=2048, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerExpertBlock(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X):
        """
        Args:
            X: sequence features [B, T_seg, D]

        Returns:
            H: transformed features [B, T_seg, D]
        """
        H = X
        for layer in self.layers:
            H = layer(H)
        return H

    def get_centroid(self, X):
        """
        Get the mean hidden representation for ELM feature similarity.
        Used for computing cosine similarity between experts.

        Args:
            X: sequence features [B, T_seg, D]

        Returns:
            centroid: mean feature vector [D]
        """
        with torch.no_grad():
            H = X
            for layer in self.layers:
                H = layer(H)
            return H.mean(dim=(0, 1))  # [D]
