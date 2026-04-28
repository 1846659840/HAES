"""
Unit tests for HAES model components.
Tests each module in isolation and integration.
"""

import torch
import pytest
import numpy as np

from models.gating import FamilyGate, ExpertGate, HierarchicalGate
from models.experts import TransformerExpert, TransformerExpertBlock
from models.hmoe import HMoE
from models.haes import HAES, AnomalyScoringHead
from models.constraints import (
    DistillationLoss,
    FeatureConsistencyLoss,
    RoutingDistillationLoss,
    EWCLoss,
    TemporalConsistencyLoss,
)
from models.elm import ActivationTracker, ExpertLifecycleManager
from utils.metrics import compute_ap, compute_auc, compute_bwt
from data.dataset import collate_variable_length
from data.incremental_split import IncrementalDataSplitter, SyntheticIncrementalStream


class TestGating:
    """Test gating networks (Section III-B, Eq. 5-6)."""

    def test_family_gate_forward(self):
        gate = FamilyGate(dim=512, num_families=4, top_k=2)
        x_bar = torch.randn(8, 512)
        idx, weights, raw = gate(x_bar)
        assert idx.shape == (8, 2)  # [B, k1]
        assert weights.shape == (8, 2)
        assert raw.shape == (8, 4)  # [B, M]
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(8))

    def test_expert_gate_forward(self):
        gate = ExpertGate(dim=512, experts_per_family=3, top_k=2)
        x_bar = torch.randn(4, 512)
        idx, weights, raw = gate(x_bar)
        assert idx.shape == (4, 2)
        assert raw.shape == (4, 3)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4))

    def test_hierarchical_gate_forward(self):
        gate = HierarchicalGate(dim=512, num_families=4,
                                experts_per_family=3, top_k1=2, top_k2=2)
        X = torch.randn(4, 20, 512)  # [B, T_seg, D]
        routing_info = gate(X)
        assert "g1_raw" in routing_info
        assert routing_info["g1_raw"].shape == (4, 4)
        assert len(routing_info["g2_selected"]) == 4

    def test_load_balance_loss(self):
        gate = HierarchicalGate(dim=512, num_families=4,
                                experts_per_family=3, top_k1=2, top_k2=2)
        g1_raw = torch.randn(4, 4).softmax(dim=-1)
        loss = gate.get_load_balance_loss(g1_raw)
        assert loss.item() >= 0


class TestExperts:
    """Test Transformer expert blocks (Section III-B, Eq. 7)."""

    def test_transformer_block_forward(self):
        block = TransformerExpertBlock(dim=512, num_heads=8, ffn_dim=2048)
        x = torch.randn(2, 16, 512)
        out = block(x)
        assert out.shape == x.shape

    def test_transformer_expert_forward(self):
        expert = TransformerExpert(dim=512, num_heads=8, ffn_dim=2048,
                                    num_layers=2)
        X = torch.randn(2, 16, 512)
        H = expert(X)
        assert H.shape == X.shape

    def test_expert_centroid(self):
        expert = TransformerExpert(dim=512, num_heads=8, ffn_dim=2048,
                                    num_layers=2)
        X = torch.randn(2, 16, 512)
        centroid = expert.get_centroid(X)
        assert centroid.shape == (512,)


class TestHMoE:
    """Test Hierarchical Mixture of Experts (Section III-B)."""

    def test_hmoe_forward(self):
        hmoe = HMoE(input_dim=512, latent_dim=512, num_families=4,
                     experts_per_family=3, top_k1=2, top_k2=2,
                     expert_num_layers=1)
        F_seq = torch.randn(4, 20, 512)
        H_fused = hmoe(F_seq)
        assert H_fused.shape == (4, 20, 512)  # [B, T_seg, D]

    def test_hmoe_with_routing(self):
        hmoe = HMoE(input_dim=512, latent_dim=512, num_families=4,
                     experts_per_family=3, top_k1=2, top_k2=2)
        F_seq = torch.randn(4, 20, 512)
        H_fused, routing = hmoe(F_seq, return_routing=True)
        assert H_fused.shape == (4, 20, 512)
        assert "g1_raw" in routing

    def test_feature_encoding(self):
        hmoe = HMoE(input_dim=512, latent_dim=512, num_families=4,
                     experts_per_family=3, top_k1=2, top_k2=2)
        F_seq = torch.randn(4, 20, 512)
        X = hmoe.encode_features(F_seq)
        assert X.shape == (4, 20, 512)

    def test_add_expert(self):
        hmoe = HMoE(input_dim=512, latent_dim=256, num_families=4,
                     experts_per_family=2, top_k1=1, top_k2=2)
        initial_count = hmoe.get_active_expert_count()
        success = hmoe.add_expert_to_family(0)
        if success:
            assert hmoe.get_active_expert_count() >= initial_count


class TestAnomalyScoring:
    """Test anomaly scoring head (Section III-D, Eq. 20)."""

    def test_scoring_forward(self):
        head = AnomalyScoringHead(latent_dim=512, num_classes=7, top_k=3)
        H_fused = torch.randn(4, 20, 512)
        outputs = head(H_fused)
        assert outputs["segment_logits"].shape == (4, 20, 7)
        assert outputs["video_logits"].shape == (4, 7)
        assert outputs["video_scores"].shape == (4, 7)
        # Scores should be in [0, 1]
        assert (outputs["video_scores"] >= 0).all()
        assert (outputs["video_scores"] <= 1).all()


class TestConstraintLosses:
    """Test constraint losses (Section III-C)."""

    def test_kd_loss(self):
        kd = DistillationLoss(temperature=4.0, top_k_class=5)
        s_logits = torch.randn(2, 16, 7)
        t_logits = torch.randn(2, 16, 7)
        loss = kd(s_logits, t_logits)
        assert loss.item() >= 0

    def test_mse_loss(self):
        mse = FeatureConsistencyLoss()
        s_feat = torch.randn(2, 16, 512)
        t_feat = torch.randn(2, 16, 512)
        loss = mse(s_feat, t_feat)
        assert loss.item() >= 0

    def test_temporal_consistency(self):
        temp = TemporalConsistencyLoss(window_size=3)
        logits = torch.randn(2, 20, 7)
        loss = temp(logits)
        assert loss.item() >= 0


class TestHAESModel:
    """Test full HAES model integration (Section III)."""

    def test_haes_forward(self):
        config = {
            "input_dim": 512, "latent_dim": 256, "num_classes": 7,
            "num_families": 4, "experts_per_family": 3,
            "top_k1": 2, "top_k2": 2, "top_k_segments": 3,
            "expert_num_heads": 4, "expert_ffn_dim": 1024,
            "expert_num_layers": 1, "dropout": 0.1, "max_seq_len": 200,
            "temperature": 4.0, "top_k_class": 5,
            "lambda_kd": 1.0, "lambda_mse": 1.0,
            "lambda_r": 1.0, "lambda_ewc": 100.0,
            "lambda_temp": 0.1, "ewc_fisher_decay": 0.9,
            "warmup_epochs": 3,
            "elm_config": {
                "add_patience": 5, "add_threshold_delta": 0.05,
                "add_family_threshold": 0.65, "merge_threshold": 0.78,
                "recycle_threshold": 0.20, "cooldown_epochs": 2,
                "warmup_epochs": 1, "recycling_interval": 5,
                "max_experts": 32, "window_size": 256,
            }
        }
        model = HAES(config)
        F_seq = torch.randn(2, 16, 512)
        outputs = model(F_seq)
        assert "video_logits" in outputs
        assert outputs["video_logits"].shape == (2, 7)

    def test_haes_loss(self):
        config = {
            "input_dim": 512, "latent_dim": 256, "num_classes": 7,
            "num_families": 4, "experts_per_family": 3,
            "top_k1": 2, "top_k2": 2, "top_k_segments": 3,
            "expert_num_heads": 4, "expert_ffn_dim": 1024,
            "expert_num_layers": 1, "dropout": 0.1, "max_seq_len": 200,
            "temperature": 4.0, "top_k_class": 5,
            "lambda_kd": 1.0, "lambda_mse": 1.0,
            "lambda_r": 1.0, "lambda_ewc": 100.0,
            "lambda_temp": 0.1, "ewc_fisher_decay": 0.9,
            "warmup_epochs": 3,
            "elm_config": {"window_size": 256, "recycle_threshold": 0.2,
                           "merge_threshold": 0.78, "add_patience": 5},
        }
        model = HAES(config)
        model.in_warmup = False  # Activate constraints
        F_seq = torch.randn(2, 16, 512)
        labels = torch.randint(0, 7, (2,))
        outputs = model(F_seq)
        total_loss, loss_dict = model.compute_loss(outputs, labels)
        assert total_loss.item() > 0
        assert "cls" in loss_dict


class TestMetrics:
    """Test evaluation metrics (Section IV-A, Eq. 25-27)."""

    def test_ap(self):
        scores = np.array([0.9, 0.8, 0.3, 0.1])
        labels = np.array([1, 1, 0, 0])
        ap = compute_ap(scores, labels)
        assert 0 <= ap <= 1

    def test_auc(self):
        scores = np.array([0.9, 0.8, 0.3, 0.1])
        labels = np.array([1, 1, 0, 0])
        auc = compute_auc(scores, labels)
        assert 0 <= auc <= 1

    def test_bwt_zero_forgetting(self):
        # Perfect retention: R_T,i = R_i,i
        perf = np.array([[0.9, 0, 0], [0.9, 0.85, 0], [0.9, 0.85, 0.8]])
        bwt = compute_bwt(perf)
        assert bwt == 0.0

    def test_bwt_negative_forgetting(self):
        # Performance degrades across phases
        perf = np.array([[0.9, 0, 0], [0.7, 0.85, 0], [0.5, 0.6, 0.8]])
        bwt = compute_bwt(perf)
        assert bwt < 0


class TestDataUtils:
    """Test data utilities."""

    def test_collate_variable_length(self):
        batch = [
            (torch.randn(10, 512), 0),
            (torch.randn(15, 512), 1),
            (torch.randn(8, 512), 2),
        ]
        padded, labels, lengths = collate_variable_length(batch)
        assert padded.shape == (3, 15, 512)  # Max length = 15
        assert torch.all(lengths == torch.tensor([10, 15, 8]))
        assert torch.all(labels == torch.tensor([0, 1, 2]))


class TestSceneDrift:
    """Test scene distribution drift simulation (Appendix C)."""

    def test_low_light_drift(self):
        stream = SyntheticIncrementalStream(base_data=None, drift_type="illumination")
        features = torch.randn(4, 16, 512)
        drifted = stream.apply_low_light(features, sigma=0.1)
        assert drifted.shape == features.shape

    def test_low_resolution_drift(self):
        stream = SyntheticIncrementalStream(base_data=None, drift_type="low_resolution")
        features = torch.randn(4, 16, 512)
        drifted = stream.apply_low_resolution(features)
        assert drifted.shape == features.shape

    def test_infrared_drift(self):
        stream = SyntheticIncrementalStream(base_data=None, drift_type="infrared")
        features = torch.randn(4, 16, 512)
        drifted = stream.apply_infrared_shift(features, channel_shift=0.15)
        assert drifted.shape == features.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
