"""
Expert Lifecycle Management (ELM) Mechanism.
Section III-E: Dynamic expert addition, merging, and recycling.

Three operations governed by activation frequency and feature similarity:
1. Addition: Insert new expert when loss stagnates & family is overloaded
2. Merging: Combine similar experts within same family
3. Recycling: Reset dormant experts (activation < threshold)

Indicators (Section III-E line 624-626):
- u_bar_e: activation frequency over sliding window W
- c_e: family-normalised mean L2 contribution magnitude (importance)
- u_tilde_e = u_bar_e * c_e: importance-weighted activation indicator (long-tail safeguard)
- rho_ij: cosine similarity between expert centroid features

Long-tail safeguards (paper line 624):
(i)  Importance-weighted indicator u_tilde_e = u_bar_e * c_e
(ii) Class-conditional protection: skip recycling experts whose top-3 activated
     classes include any class with empirical frequency below the 10th percentile
(iii) E_cooldown = 10 epoch cooldown after structural changes
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque, defaultdict


class ActivationTracker:
    """
    Tracks expert activation frequency, L2 contribution magnitude, and
    per-class activation history over a sliding window of W batches.
    """

    def __init__(self, window_size=2048):
        self.window_size = window_size
        self.activation_counts = {}  # (family, expert) -> deque of activation flags
        self.contribution_norms = {}  # (family, expert) -> deque of L2 norms (for c_e)
        self.class_activations = {}  # (family, expert) -> deque of class indices
        self.label_history = deque(maxlen=window_size)  # rolling class label log
        self.total_batches = 0

    def record_batch(self, routing_info, contribution_map=None, labels=None):
        """
        Record which experts were activated in this batch, plus their L2
        contribution magnitudes and the labels they processed.

        Args:
            routing_info: dict from HierarchicalGate.forward()
            contribution_map: optional dict {(m, n): float} mapping expert key
                to its mean L2 contribution magnitude in this batch (for c_e).
            labels: optional [B] tensor of true class indices, used to populate
                the per-(expert, class) co-activation log for rare-class
                protection during recycling.
        """
        self.total_batches += 1
        batch_activated = set()
        contribution_map = contribution_map or {}

        if labels is not None:
            for y in labels.detach().cpu().numpy().tolist():
                self.label_history.append(int(y))

        for b in range(len(routing_info["g2_selected"])):
            label_b = None
            if labels is not None and b < labels.shape[0]:
                label_b = int(labels[b].item())

            for m_local, (family_idx_tensor, _) in enumerate(
                zip(routing_info["g1_selected_idx"][b],
                     routing_info["g1_selected_weights"][b])
            ):
                m = family_idx_tensor.item()
                experts = routing_info["g2_selected"][b][m_local][0].squeeze(0)
                for n_tensor in experts:
                    n = n_tensor.item()
                    key = (m, n)
                    batch_activated.add(key)
                    if key not in self.activation_counts:
                        self.activation_counts[key] = deque(maxlen=self.window_size)
                        self.contribution_norms[key] = deque(maxlen=self.window_size)
                        self.class_activations[key] = deque(maxlen=self.window_size)
                    self.activation_counts[key].append(1)
                    if key in contribution_map:
                        self.contribution_norms[key].append(contribution_map[key])
                    if label_b is not None:
                        self.class_activations[key].append(label_b)

        for key in self.activation_counts:
            if key not in batch_activated:
                self.activation_counts[key].append(0)

    def get_activation_frequency(self, m, n):
        """u_bar_e: fraction of batches where expert was activated."""
        key = (m, n)
        if key not in self.activation_counts or len(self.activation_counts[key]) == 0:
            return 0.0
        return sum(self.activation_counts[key]) / len(self.activation_counts[key])

    def get_family_mean_activation(self, m, expert_count):
        """Mean activation frequency across all experts in family m."""
        freqs = [self.get_activation_frequency(m, n) for n in range(expert_count)]
        if not freqs:
            return 0.0
        return np.mean(freqs)

    def get_mean_contribution(self, m, n):
        """Mean L2 contribution magnitude of expert (m, n) over the window."""
        key = (m, n)
        if key not in self.contribution_norms or len(self.contribution_norms[key]) == 0:
            return 0.0
        return float(np.mean(list(self.contribution_norms[key])))

    def get_family_mean_contribution(self, m, expert_count, exclude_n=None):
        """Family-level mean of per-expert mean contribution magnitudes.
        `exclude_n` removes the expert under self-normalisation to avoid the
        self-inclusion bias in c_e (Section III-E line 624)."""
        contribs = [
            self.get_mean_contribution(m, n)
            for n in range(expert_count)
            if (exclude_n is None or n != exclude_n)
        ]
        contribs = [c for c in contribs if c > 0]
        if not contribs:
            return 0.0
        return float(np.mean(contribs))

    def get_normalised_contribution(self, m, n, expert_count):
        """c_e: family-normalised mean L2 contribution magnitude.
        Family mean excludes the expert itself so an outlier does not pull
        its own normaliser toward its own value (paper line 624 safeguard i)."""
        family_mean = self.get_family_mean_contribution(
            m, expert_count, exclude_n=n
        )
        if family_mean <= 1e-8:
            return 1.0  # Neutral when no peer contribution data yet.
        return self.get_mean_contribution(m, n) / family_mean

    def get_importance_weighted_indicator(self, m, n, expert_count):
        """u_tilde_e = u_bar_e * c_e (paper line 624 safeguard i)."""
        return (
            self.get_activation_frequency(m, n)
            * self.get_normalised_contribution(m, n, expert_count)
        )

    def get_top_classes(self, m, n, k=3):
        """Most frequently co-activated class indices for expert (m, n)."""
        key = (m, n)
        if key not in self.class_activations or len(self.class_activations[key]) == 0:
            return []
        counter = defaultdict(int)
        for cls in self.class_activations[key]:
            counter[cls] += 1
        return [c for c, _ in sorted(counter.items(), key=lambda x: -x[1])[:k]]

    def get_rare_class_set(self, percentile=10, num_classes=None):
        """Classes whose empirical frequency falls below the `percentile`-th
        percentile. When `num_classes` is provided, the percentile is computed
        over the full label vocabulary (classes with zero observed frequency
        are included), so unseen classes are correctly recognised as the
        rarest (paper line 624 safeguard ii)."""
        counter = defaultdict(int)
        for cls in self.label_history:
            counter[cls] += 1
        if num_classes is not None:
            for c in range(num_classes):
                counter.setdefault(c, 0)
        if not counter:
            return set()
        freqs = np.array(list(counter.values()), dtype=np.float64)
        threshold = np.percentile(freqs, percentile)
        return {cls for cls, f in counter.items() if f <= threshold}

    def get_all_activations(self):
        """Get all tracked activation frequencies."""
        result = {}
        for (m, n), counts in self.activation_counts.items():
            if len(counts) > 0:
                result[(m, n)] = sum(counts) / len(counts)
        return result


class FeatureSimilarityTracker:
    """Tracks expert feature centroids for merging decisions."""

    def __init__(self):
        self.centroids = {}  # (m, n) -> centroid vector
        self.centroid_counts = {}  # (m, n) -> number of accumulated samples

    def update_centroid(self, m, n, new_centroid, alpha=0.9):
        """Exponential moving average update of expert centroid."""
        key = (m, n)
        if key in self.centroids:
            self.centroids[key] = (
                alpha * self.centroids[key] + (1 - alpha) * new_centroid
            )
            self.centroid_counts[key] += 1
        else:
            self.centroids[key] = new_centroid
            self.centroid_counts[key] = 1

    def get_cosine_similarity(self, m, i, j):
        """rho_ij: cosine similarity between expert i and j in family m."""
        key_i = (m, i)
        key_j = (m, j)
        if key_i not in self.centroids or key_j not in self.centroids:
            return 0.0
        c_i = self.centroids[key_i]
        c_j = self.centroids[key_j]
        cos_sim = torch.nn.functional.cosine_similarity(
            c_i.unsqueeze(0), c_j.unsqueeze(0)
        )
        return cos_sim.item()


class ExpertLifecycleManager:
    """
    ELM: Governs Add / Merge / Recycle of experts across incremental phases.

    Decision thresholds (paper-wide constants, not tuned per-dataset; Section IV-B line 704):
    - tau_u^add = 0.65 (family activation overload)
    - tau_l = 0.20 (recycling threshold on u_tilde_e)
    - delta_merge = 0.92 (merging similarity threshold)
    - delta_add = 0.05 (loss stagnation)
    - E_patience = 5 epochs
    - E_cooldown = 10 epochs
    - E_warmup = 5 epochs
    - sigma_init^2 = 0.02 (variance for recycled experts)
    """

    def __init__(self, config, hmoe_model, device="cuda"):
        self.config = config
        self.model = hmoe_model
        self.device = device

        # Thresholds from config
        self.add_patience = config.get("add_patience", 5)
        self.add_delta = config.get("add_threshold_delta", 0.05)
        self.add_family_threshold = config.get("add_family_threshold", 0.65)
        self.merge_threshold = config.get("merge_threshold", 0.92)
        self.recycle_threshold = config.get("recycle_threshold", 0.20)
        self.cooldown_epochs = config.get("cooldown_epochs", 10)
        # ELM warmup matches trainer warmup (5 epochs per Section IV-B line 708)
        self.warmup_epochs = config.get("warmup_epochs", 5)
        self.recycling_interval = config.get("recycling_interval", 5)
        self.max_experts = config.get("max_experts", 32)
        self.window_size = config.get("window_size", 2048)
        # Variance for recycled-expert reinit per Section IV-B (sigma_init^2 = 0.02)
        self.init_var = config.get("init_var", config.get("init_std", 0.02))
        # Long-tail safeguard percentile (paper line 624: 10th percentile)
        self.rare_class_percentile = config.get("rare_class_percentile", 10)
        self.rare_class_topk = config.get("rare_class_topk", 3)
        # Updated by trainer.start_phase to enumerate the full label vocabulary
        # for the rare-class percentile (so unseen classes are protected).
        self.num_classes = None

        # Trackers
        self.activation_tracker = ActivationTracker(window_size=self.window_size)
        self.similarity_tracker = FeatureSimilarityTracker()

        # State
        self.epochs_since_add = 0
        self.epochs_since_structural_change = 0
        self.phase_epoch = 0
        self.current_phase = 0
        self.loss_history = []
        self.total_additions = 0
        self.total_merges = 0
        self.total_recycles = 0

    def record_batch(self, routing_info, X, expert_outputs=None, labels=None):
        """
        Record routing activations, contribution magnitudes (for c_e), and
        per-(expert, class) co-activation (for rare-class protection).

        Args:
            routing_info: gating decisions dict.
            X: encoded clip features [B, T_seg, D] used for centroid updates.
            expert_outputs: optional dict {(m, n): tensor} mapping each
                activated expert key to its output tensor for this batch
                (used to compute the c_e contribution magnitude). When None,
                a fresh forward pass is used.
            labels: optional [B] true class indices for rare-class tracking.
        """
        contribution_map = {}

        # Centroid updates and per-expert contribution magnitudes.
        for b in range(len(routing_info["g2_selected"])):
            for m_local, (family_idx_tensor, _) in enumerate(
                zip(routing_info["g1_selected_idx"][b],
                     routing_info["g1_selected_weights"][b])
            ):
                m = family_idx_tensor.item()
                experts = routing_info["g2_selected"][b][m_local][0].squeeze(0)
                for n_tensor in experts:
                    n = n_tensor.item()
                    key = (m, n)
                    try:
                        with torch.no_grad():
                            out = (
                                expert_outputs[key]
                                if expert_outputs and key in expert_outputs
                                else self.model.experts[m][n](X[b:b+1])
                            )
                            self.similarity_tracker.update_centroid(
                                m, n, out.mean(dim=(0, 1))
                            )
                            contribution_map[key] = float(
                                out.norm(p=2, dim=-1).mean().item()
                            )
                    except (IndexError, RuntimeError):
                        pass

        self.activation_tracker.record_batch(
            routing_info, contribution_map=contribution_map, labels=labels
        )

    def step_epoch(self, current_loss):
        """
        Called at the end of each epoch. Decides structural changes.

        Returns:
            action: str describing action taken, or None
        """
        self.phase_epoch += 1
        self.loss_history.append(current_loss)

        # Warm-up: skip structural changes
        if self.phase_epoch <= self.warmup_epochs:
            return None

        # Cooldown: wait after last structural change
        if self.epochs_since_structural_change < self.cooldown_epochs:
            self.epochs_since_structural_change += 1
            return None

        # Check for Expert Addition (only in incremental phases, phase > 0)
        if self.current_phase > 0 and self._should_add_expert():
            return self._add_expert()

        # Check for Expert Merging (only in incremental phases)
        if self.current_phase > 0 and self._should_merge():
            return self._merge_experts()

        # Check for Expert Recycling (checked every epoch)
        if self.current_phase > 0 and self.phase_epoch % self.recycling_interval == 0:
            recycle_result = self._recycle_experts()
            if recycle_result:
                return recycle_result

        return None

    def _should_add_expert(self):
        """
        Trigger addition when L_cls fails to decrease by more than delta_add
        across the last E_patience consecutive epochs (paper Section III-E
        line 599-602). No further accumulator — a single E_patience window of
        stagnation is sufficient.
        """
        if len(self.loss_history) < self.add_patience:
            return False

        recent_losses = self.loss_history[-self.add_patience:]
        loss_change = abs(recent_losses[0] - recent_losses[-1])
        if loss_change > self.add_delta:
            return False

        if self.model.get_active_expert_count() >= self.max_experts:
            return False

        family_sizes = self.model.get_family_sizes()
        family_activations = [
            self.activation_tracker.get_family_mean_activation(m, size)
            for m, size in enumerate(family_sizes)
        ]
        if not family_activations or max(family_activations) < self.add_family_threshold:
            return False

        return True

    def _add_expert(self):
        """Add a new expert to the most overloaded family."""
        family_sizes = self.model.get_family_sizes()
        family_activations = [
            self.activation_tracker.get_family_mean_activation(m, size)
            for m, size in enumerate(family_sizes)
        ]
        m_star = int(np.argmax(family_activations))

        success = self.model.add_expert_to_family(m_star)
        if success:
            self.total_additions += 1
            self.epochs_since_structural_change = 0
            return f"Added expert to family {m_star} (total experts: {self.model.get_active_expert_count()})"
        return None

    def _should_merge(self):
        """Check if any expert pair should be merged."""
        family_sizes = self.model.get_family_sizes()
        for m, size in enumerate(family_sizes):
            for i in range(size):
                for j in range(i + 1, size):
                    sim = self.similarity_tracker.get_cosine_similarity(m, i, j)
                    if sim >= self.merge_threshold:
                        return True
        return False

    def _merge_experts(self):
        """Merge the most similar expert pair."""
        best_sim = -1
        best_pair = None

        family_sizes = self.model.get_family_sizes()
        for m, size in enumerate(family_sizes):
            for i in range(size):
                for j in range(i + 1, size):
                    sim = self.similarity_tracker.get_cosine_similarity(m, i, j)
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (m, i, j)

        if best_pair and best_sim >= self.merge_threshold:
            m, i, j = best_pair
            alpha_i = self.activation_tracker.get_activation_frequency(m, i)
            alpha_j = self.activation_tracker.get_activation_frequency(m, j)
            success = self.model.merge_experts(m, i, j, max(alpha_i, 0.01),
                                               max(alpha_j, 0.01))
            if success:
                self.total_merges += 1
                self.epochs_since_structural_change = 0
                if (m, i) in self.similarity_tracker.centroids and (m, j) in self.similarity_tracker.centroids:
                    self.similarity_tracker.centroids[(m, i)] = (
                        alpha_i * self.similarity_tracker.centroids[(m, i)] +
                        alpha_j * self.similarity_tracker.centroids[(m, j)]
                    ) / (alpha_i + alpha_j + 1e-8)
                self.similarity_tracker.centroids.pop((m, j), None)
                self.similarity_tracker.centroid_counts.pop((m, j), None)
                self.activation_tracker.activation_counts.pop((m, j), None)
                self.activation_tracker.contribution_norms.pop((m, j), None)
                self.activation_tracker.class_activations.pop((m, j), None)
                return f"Merged experts {i},{j} in family {m} (sim={best_sim:.3f})"
        return None

    def _recycle_experts(self):
        """
        Recycle dormant experts using the importance-weighted indicator
        u_tilde_e = u_bar_e * c_e (Section III-E safeguard i) and skip
        experts whose top-3 activated classes intersect the rare-class set
        (Section III-E safeguard ii).
        """
        recycled = []
        family_sizes = self.model.get_family_sizes()
        rare_classes = self.activation_tracker.get_rare_class_set(
            percentile=self.rare_class_percentile,
            num_classes=self.num_classes,
        )

        for m, size in enumerate(family_sizes):
            for n in range(size):
                u_tilde = self.activation_tracker.get_importance_weighted_indicator(
                    m, n, size
                )
                if u_tilde >= self.recycle_threshold or size <= 2:
                    continue

                top_classes = self.activation_tracker.get_top_classes(
                    m, n, k=self.rare_class_topk
                )
                if rare_classes and any(c in rare_classes for c in top_classes):
                    # Class-conditional protection: skip rare-class specialists.
                    continue

                self.model.recycle_expert(m, n, init_var=self.init_var)
                recycled.append(
                    f"family{m}_expert{n} (u_tilde={u_tilde:.4f})"
                )
                self.total_recycles += 1
                self.activation_tracker.activation_counts.pop((m, n), None)
                self.activation_tracker.contribution_norms.pop((m, n), None)
                self.activation_tracker.class_activations.pop((m, n), None)
                self.activation_tracker.activation_counts[(m, n)] = deque(
                    maxlen=self.window_size
                )
                self.activation_tracker.contribution_norms[(m, n)] = deque(
                    maxlen=self.window_size
                )
                self.activation_tracker.class_activations[(m, n)] = deque(
                    maxlen=self.window_size
                )
                self.similarity_tracker.centroids.pop((m, n), None)
                self.similarity_tracker.centroid_counts.pop((m, n), None)

        if recycled:
            self.epochs_since_structural_change = 0
            return f"Recycled {len(recycled)} experts: {', '.join(recycled)}"
        return None

    def start_phase(self, phase_idx, num_classes=None):
        """Initialize tracking for a new incremental phase. `num_classes`
        is forwarded to the rare-class percentile so unseen classes are
        included in the rare set (paper line 624 safeguard ii)."""
        self.current_phase = phase_idx
        self.phase_epoch = 0
        self.loss_history = []
        self.epochs_since_structural_change = self.cooldown_epochs
        if num_classes is not None:
            self.num_classes = num_classes

    def get_stats(self):
        """Return ELM statistics."""
        return {
            "active_experts": self.model.get_active_expert_count(),
            "family_sizes": self.model.get_family_sizes(),
            "total_additions": self.total_additions,
            "total_merges": self.total_merges,
            "total_recycles": self.total_recycles,
            "activation_frequencies": self.activation_tracker.get_all_activations(),
        }
