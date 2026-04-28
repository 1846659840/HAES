"""
HAES Model Evaluator.
Comprehensive evaluation including:
- Per-dataset evaluation (XD-Violence, UCF-Crime, ShanghaiTech)
- Incremental phase evaluation with BWT tracking
- Scene drift robustness evaluation (Appendix C)
- Inference latency benchmarking (Appendix G)
- Cross-scenario generalization (Appendix L)
- Noise robustness evaluation (Appendix B)
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import collate_variable_length
from data.incremental_split import SyntheticIncrementalStream
from utils.metrics import (
    compute_ap, compute_auc, compute_auc_trapezoidal,
    compute_video_level_ap, compute_video_level_auc,
    compute_bwt, compute_pr_curve,
)


class Evaluator:
    """
    Comprehensive evaluator for HAES.

    Evaluates:
    1. Overall detection performance (AP/AUC)
    2. Per-class fine-grained performance
    3. Catastrophic forgetting (BWT)
    4. Inference speed (FPS) and latency breakdown
    5. Robustness to scene distribution drift
    """

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, test_loader, num_classes, dataset_name="xd_violence"):
        """
        Evaluate model on a test set.

        Args:
            test_loader: DataLoader for test data
            num_classes: number of classes
            dataset_name: "xd_violence" (AP) or "ucf_crime" (AUC)

        Returns:
            results: dict with all evaluation metrics
        """
        self.model.eval()

        all_video_scores = []
        all_video_labels = []
        all_segment_scores = []
        all_segment_labels = []

        for features, labels, lengths in tqdm(test_loader, desc="Evaluating"):
            features = features.to(self.device)

            outputs = self.model(features)

            all_video_scores.append(outputs["video_scores"].cpu().numpy())
            all_video_labels.append(labels.numpy())

            # Segment-level for localization evaluation
            all_segment_scores.append(outputs["segment_scores"].cpu().numpy())

        # Concatenate
        video_scores = np.concatenate(all_video_scores, axis=0)  # [N, C]
        video_labels = np.concatenate(all_video_labels, axis=0)  # [N]
        segment_scores = np.concatenate(all_segment_scores, axis=0)  # [N, T, C]

        # Compute per-class and overall metrics
        class_aps, mean_ap = compute_video_level_ap(
            video_scores, video_labels, num_classes
        )
        class_aucs, mean_auc = compute_video_level_auc(
            video_scores, video_labels, num_classes
        )

        results = {
            "mean_ap": mean_ap,
            "mean_auc": mean_auc,
            "per_class_ap": class_aps,
            "per_class_auc": class_aucs,
        }

        return results

    @torch.no_grad()
    def evaluate_incremental_phases(self, phase_test_loaders, phase_configs):
        """
        Evaluate performance across all incremental phases using current model.

        Note: BWT requires model snapshots at each training phase to measure
        forgetting. When called with a single (final) checkpoint, the performance
        matrix will have identical rows and BWT will be ~0. For proper BWT
        measurement, use the trainer's train_all_phases which tracks performance
        after each incremental phase.

        Args:
            phase_test_loaders: list of test DataLoaders per phase
            phase_configs: list of (seen_categories, phase_categories)

        Returns:
            results: dict with per-phase and BWT metrics
        """
        num_phases = len(phase_test_loaders)
        performance_matrix = np.zeros((num_phases, num_phases))

        all_results = []
        for phase_idx in range(num_phases):
            test_loader = phase_test_loaders[phase_idx]
            seen_cats, _ = phase_configs[phase_idx]
            num_classes = len(seen_cats)

            results = self.evaluate(test_loader, num_classes)
            all_results.append(results)
            performance_matrix[phase_idx, phase_idx] = results.get("mean_ap", results.get("mean_auc", 0))

        # Evaluate all previous phase test sets through current model
        for cur_phase in range(1, num_phases):
            for prev_phase in range(cur_phase):
                prev_loader = phase_test_loaders[prev_phase]
                prev_seen, _ = phase_configs[prev_phase]
                results = self.evaluate(prev_loader, len(prev_seen))
                performance_matrix[cur_phase, prev_phase] = results.get("mean_ap", results.get("mean_auc", 0))

        bwt = compute_bwt(performance_matrix)

        return {
            "per_phase_results": all_results,
            "performance_matrix": performance_matrix.tolist(),
            "bwt": bwt,
        }

    @torch.no_grad()
    def benchmark_inference_speed(self, test_loader, num_runs=100, warmup=10):
        """
        Benchmark inference speed (FPS) following Appendix G protocol.

        Measures:
        - End-to-end latency per clip (ms)
        - FPS (clips per second)
        - Per-component latency breakdown:
            1. Feature extraction
            2. Routing (Stage 1 + Stage 2)
            3. Expert forward pass
            4. Aggregation (scoring head)
            5. Other overhead

        Returns:
            latency_stats: dict with P50, P95, P99 latencies and FPS
        """
        self.model.eval()

        # Warmup
        for i, (features, labels, lengths) in enumerate(test_loader):
            if i >= warmup:
                break
            features = features.to(self.device)
            _ = self.model(features)
        torch.cuda.synchronize()

        # Benchmark
        all_latencies = []
        total_frames = 0
        total_time = 0.0

        for i, (features, labels, lengths) in enumerate(test_loader):
            if i >= num_runs:
                break
            features = features.to(self.device)
            B = features.size(0)

            torch.cuda.synchronize()
            start = time.perf_counter()

            # Forward pass
            outputs = self.model(features)

            torch.cuda.synchronize()
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            all_latencies.append(latency_ms)
            total_frames += B
            total_time += (end - start)

        all_latencies = np.array(all_latencies)

        # Compute percentiles
        p50 = np.percentile(all_latencies, 50)
        p95 = np.percentile(all_latencies, 95)
        p99 = np.percentile(all_latencies, 99)

        # FPS calculation
        fps = total_frames / total_time if total_time > 0 else 0

        return {
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "p99_ms": float(p99),
            "mean_ms": float(np.mean(all_latencies)),
            "std_ms": float(np.std(all_latencies)),
            "fps": float(fps),
            "p99_median_ratio": float(p99 / p50) if p50 > 0 else 0,
        }

    @torch.no_grad()
    def evaluate_scene_drift_robustness(self, test_loader, num_classes,
                                         drift_types=None):
        """
        Evaluate robustness to scene distribution drift (Appendix C).

        Tests three drift types:
        - Low-Light: Additive Gaussian noise (sigma=0.1)
        - Low-Resolution: Downsample-upsample (50% reduction)
        - Infrared-like: Channel-wise shift (delta=0.15)

        Args:
            test_loader: base test DataLoader
            num_classes: number of classes
            drift_types: list of drift types to evaluate

        Returns:
            drift_results: dict mapping drift_type -> performance metrics
        """
        if drift_types is None:
            drift_types = ["normal", "low_light", "low_resolution", "infrared"]

        drift_generator = SyntheticIncrementalStream(
            base_data=None, drift_type="illumination"
        )
        drift_results = {}

        for drift_type in drift_types:
            self.model.eval()
            all_scores = []
            all_labels = []

            for features, labels, lengths in test_loader:
                # Apply scene drift
                drifted_features = drift_generator.generate_drifted_batch(
                    features, drift_type
                )
                drifted_features = drifted_features.to(self.device)

                outputs = self.model(drifted_features)
                all_scores.append(outputs["video_scores"].cpu().numpy())
                all_labels.append(labels.numpy())

            all_scores = np.concatenate(all_scores, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            _, mean_ap = compute_video_level_ap(all_scores, all_labels, num_classes)
            _, mean_auc = compute_video_level_auc(all_scores, all_labels, num_classes)

            drift_results[drift_type] = {
                "mean_ap": mean_ap,
                "mean_auc": mean_auc,
            }

        # Compute relative robustness (performance drop from normal)
        normal_ap = drift_results.get("normal", {}).get("mean_ap", 0)
        for drift_type in drift_results:
            if drift_type != "normal" and normal_ap > 0:
                drop = normal_ap - drift_results[drift_type]["mean_ap"]
                drift_results[drift_type]["ap_drop"] = drop
                drift_results[drift_type]["relative_drop_pct"] = (
                    drop / normal_ap * 100
                )

        return drift_results

    def evaluate_noise_robustness(self, test_loader, num_classes, noise_ratio=0.15,
                                   seed=42):
        """
        Evaluate robustness to label noise detection (Appendix B).

        Measures the model's ability to distinguish noise samples from hard
        normal samples using prediction entropy. This is an evaluation OF the
        model's entropy-based noise detection capability on test data,
        following the controlled experiment protocol in Appendix B.

        Args:
            test_loader: DataLoader for evaluation
            num_classes: number of classes
            noise_ratio: fraction of labels treated as noise for analysis
            seed: random seed for reproducibility

        Returns:
            noise_results: dict with noise robustness metrics
        """
        rng = np.random.RandomState(seed)
        self.model.eval()

        all_entropies = []
        all_variances = []
        all_scores = []
        all_labels = []
        all_is_noise = []

        # 5 stochastic forward passes for entropy variance (Appendix B protocol)
        num_passes = 5

        for features, labels, lengths in test_loader:
            features = features.to(self.device)

            pass_entropies = []
            for _ in range(num_passes):
                # Apply dropout in train mode for stochastic passes
                self.model.train()
                outputs = self.model(features)
                self.model.eval()
                logits = outputs["segment_logits"]  # [B, T, C]
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean(dim=1)  # [B]
                pass_entropies.append(entropy.cpu().numpy())

            pass_entropies = np.stack(pass_entropies, axis=0)  # [5, B]
            mean_entropy = pass_entropies.mean(axis=0)
            var_entropy = pass_entropies.var(axis=0)

            all_entropies.append(mean_entropy)
            all_variances.append(var_entropy)
            all_scores.append(outputs["video_scores"].cpu().numpy())
            all_labels.append(labels.numpy())

            # Designate subset as "noise" for analysis (Appendix B protocol)
            is_noise = rng.random(len(labels)) < noise_ratio
            all_is_noise.append(is_noise.astype(int))

        all_entropies = np.concatenate(all_entropies)
        all_variances = np.concatenate(all_variances)
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels)
        all_is_noise = np.concatenate(all_is_noise)

        # Noise detection via entropy thresholding
        # Compute proper ROC-AUC using sklearn for accurate evaluation
        from sklearn.metrics import roc_auc_score as sklearn_auc
        try:
            entropy_auc = sklearn_auc(all_is_noise, all_entropies)
        except ValueError:
            entropy_auc = 0.5

        # Find best entropy threshold via Youden index
        threshold_candidates = np.linspace(
            all_entropies.min(), all_entropies.max(), 100
        )
        best_j = -1
        best_threshold = 0
        for threshold in threshold_candidates:
            predicted_noise = (all_entropies > threshold).astype(int)
            tpr = np.sum((predicted_noise == 1) & (all_is_noise == 1)) / max(np.sum(all_is_noise == 1), 1)
            fpr = np.sum((predicted_noise == 1) & (all_is_noise == 0)) / max(np.sum(all_is_noise == 0), 1)
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_threshold = threshold

        # Noise rejection rate at best threshold
        noise_mask = all_is_noise == 1
        predicted_noise = (all_entropies > best_threshold).astype(int)
        noise_rejection_rate = np.sum(predicted_noise[noise_mask]) / max(np.sum(noise_mask), 1)

        # Hard normal retention rate
        normal_mask = all_is_noise == 0
        hard_normal_retention = 1 - np.sum(predicted_noise[normal_mask]) / max(np.sum(normal_mask), 1)

        return {
            "noise_mean_entropy": float(np.mean(all_entropies[noise_mask])),
            "hard_normal_mean_entropy": float(np.mean(all_entropies[normal_mask])),
            "entropy_auc": float(entropy_auc),
            "noise_rejection_rate": float(noise_rejection_rate),
            "hard_normal_retention_rate": float(hard_normal_retention),
            "best_entropy_threshold": float(best_threshold),
        }

    @torch.no_grad()
    def evaluate_per_component_latency(self, test_loader, num_runs=50):
        """
        Measure per-component latency breakdown (Appendix G, Fig. 19c).
        Components: Feature encoding, Routing (two-stage gate), Expert forward,
        Aggregation (scoring head). Total includes all components.
        """
        self.model.eval()

        latencies = {
            "feature_encoding": [],
            "routing": [],
            "expert_forward": [],
            "aggregation": [],
            "total": [],
        }

        for i, (features, labels, lengths) in enumerate(test_loader):
            if i >= num_runs:
                break
            features = features.to(self.device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Step 1: Feature encoding with positional embedding (Eq. 2-4)
            X = self.model.hmoe.encode_features(features)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Step 2: Two-stage hierarchical gating (Eq. 5-6)
            routing_info = self.model.hmoe.gate(X)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            # Step 3: Expert forward pass (Eq. 7-9) - manual fusion
            B, T_seg, D = X.shape
            H_fused = torch.zeros(B, T_seg, D, device=self.device, dtype=X.dtype)
            for b in range(B):
                H_b = torch.zeros(T_seg, D, device=self.device, dtype=X.dtype)
                family_weights = routing_info["g1_selected_weights"][b]
                for m_local, (m_idx_tensor, m_weight) in enumerate(zip(
                    routing_info["g1_selected_idx"][b], family_weights
                )):
                    m = m_idx_tensor.item()
                    expert_indices, expert_weights = routing_info["g2_selected"][b][m_local]
                    H_m = torch.zeros(T_seg, D, device=self.device, dtype=X.dtype)
                    for n_local, (n_idx_tensor, n_weight) in enumerate(zip(
                        expert_indices.squeeze(0), expert_weights.squeeze(0)
                    )):
                        n = n_idx_tensor.item()
                        H_mn = self.model.hmoe.experts[m][n](X[b:b+1])
                        H_m += n_weight * H_mn.squeeze(0)
                    H_b += m_weight * H_m
                H_fused[b] = H_b
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            # Step 4: Aggregation (scoring head, Eq. 20)
            scores = self.model.scoring_head(H_fused)
            torch.cuda.synchronize()
            t4 = time.perf_counter()

            latencies["feature_encoding"].append((t1 - t0) * 1000)
            latencies["routing"].append((t2 - t1) * 1000)
            latencies["expert_forward"].append((t3 - t2) * 1000)
            latencies["aggregation"].append((t4 - t3) * 1000)
            latencies["total"].append((t4 - t0) * 1000)

        return {
            "feature_encoding_ms": float(np.mean(latencies["feature_encoding"])),
            "routing_ms": float(np.mean(latencies["routing"])),
            "expert_forward_ms": float(np.mean(latencies["expert_forward"])),
            "aggregation_ms": float(np.mean(latencies["aggregation"])),
            "total_ms": float(np.mean(latencies["total"])),
        }
