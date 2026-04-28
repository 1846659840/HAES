"""
Expert Lifecycle Management (ELM) Mechanism.
Section III-E: Dynamic expert addition, merging, and recycling.

Three operations governed by activation frequency and feature similarity:
1. Addition: Insert new expert when loss stagnates & family is overloaded
2. Merging: Combine similar experts within same family
3. Recycling: Reset dormant experts (activation < threshold)

Indicators:
- u_bar_e: activation frequency over sliding window W
- rho_ij: cosine similarity between expert centroid features
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque


class ActivationTracker:
    """Tracks expert activation frequency over a sliding window of W batches."""

    def __init__(self, window_size=2048):
        self.window_size = window_size
        self.activation_counts = {}  # (family, expert) -> deque of activation flags
        self.total_batches = 0

    def record_batch(self, routing_info):
        """Record which experts were activated in this batch."""
        self.total_batches += 1
        batch_activated = set()  # Track which keys were activated this batch

        for b in range(len(routing_info["g2_selected"])):
            for m_local, (family_idx_tensor, _) in enumerate(
                zip(routing_info["g1_selected_idx"][b],
                     routing_info["g1_selected_weights"][b])
            ):
                m = family_idx_tensor.item()
                experts = routing_info["g2_selected"][b][m_local][0].squeeze(0)  # [k2]
                for n_tensor in experts:
                    n = n_tensor.item()
                    key = (m, n)
                    batch_activated.add(key)
                    if key not in self.activation_counts:
                        self.activation_counts[key] = deque(maxlen=self.window_size)
                    self.activation_counts[key].append(1)

        # Only append 0 for tracked experts NOT activated in this batch
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

    Decision thresholds (paper-wide constants, not tuned per-dataset):
    - tau_u^add = 0.65 (family activation overload)
    - tau_l = 0.20 (recycling threshold)
    - tau_s = 0.78 (merging similarity threshold)
    - delta_add = 0.05 (loss stagnation)
    - E_patience = 5 epochs
    - E_cooldown = 2 epochs
    - E_warmup = 1 epoch
    """

    def __init__(self, config, hmoe_model, device="cuda"):
        self.config = config
        self.model = hmoe_model
        self.device = device

        # Thresholds from config
        self.add_patience = config.get("add_patience", 5)
        self.add_delta = config.get("add_threshold_delta", 0.05)
        self.add_family_threshold = config.get("add_family_threshold", 0.65)
        self.merge_threshold = config.get("merge_threshold", 0.78)
        self.recycle_threshold = config.get("recycle_threshold", 0.20)
        self.cooldown_epochs = config.get("cooldown_epochs", 2)
        # ELM warmup matches trainer warmup (3 epochs per Section IV-B)
        self.warmup_epochs = config.get("warmup_epochs", 3)
        self.recycling_interval = config.get("recycling_interval", 5)
        self.max_experts = config.get("max_experts", 32)
        self.window_size = config.get("window_size", 2048)

        # Trackers
        self.activation_tracker = ActivationTracker(window_size=self.window_size)
        self.similarity_tracker = FeatureSimilarityTracker()

        # State
        self.epochs_since_add = 0
        self.epochs_since_structural_change = 0
        self.phase_epoch = 0
        self.current_phase = 0
        self.loss_history = []
        self.stagnation_counter = 0
        self.total_additions = 0
        self.total_merges = 0
        self.total_recycles = 0

    def record_batch(self, routing_info, X):
        """Record routing activations and update centroids."""
        self.activation_tracker.record_batch(routing_info)

        # Update centroids for ELM
        for b in range(len(routing_info["g2_selected"])):
            for m_local, (family_idx_tensor, _) in enumerate(
                zip(routing_info["g1_selected_idx"][b],
                     routing_info["g1_selected_weights"][b])
            ):
                m = family_idx_tensor.item()
                experts = routing_info["g2_selected"][b][m_local][0].squeeze(0)
                for n_tensor in experts:
                    n = n_tensor.item()
                    try:
                        centroid = self.model.experts[m][n].get_centroid(
                            X[b:b+1]
                        )
                        self.similarity_tracker.update_centroid(m, n, centroid)
                    except (IndexError, RuntimeError):
                        pass

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

        # Check for Expert Addition
        if self._should_add_expert():
            return self._add_expert()

        # Check for Expert Merging
        if self._should_merge():
            return self._merge_experts()

        # Check for Expert Recycling (every f phases)
        if (self.current_phase > 0 and
            self.current_phase % self.recycling_interval == 0 and
            self.phase_epoch == self.warmup_epochs + 1):
            return self._recycle_experts()

        return None

    def _should_add_expert(self):
        """Check if a new expert should be added."""
        # Check loss stagnation
        if len(self.loss_history) < self.add_patience:
            return False

        recent_losses = self.loss_history[-self.add_patience:]
        loss_decrease = recent_losses[0] - recent_losses[-1]
        if loss_decrease > self.add_delta:
            self.stagnation_counter = 0
            return False

        self.stagnation_counter += 1
        if self.stagnation_counter < self.add_patience:
            return False

        # Check total expert cap
        if self.model.get_active_expert_count() >= self.max_experts:
            return False

        # Find most overloaded family
        family_sizes = self.model.get_family_sizes()
        family_activations = [
            self.activation_tracker.get_family_mean_activation(m, size)
            for m, size in enumerate(family_sizes)
        ]

        if max(family_activations) < self.add_family_threshold:
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
            self.stagnation_counter = 0
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
                # Clean up tracking metadata for removed expert j
                self.similarity_tracker.centroids.pop((m, j), None)
                self.similarity_tracker.centroid_counts.pop((m, j), None)
                self.activation_tracker.activation_counts.pop((m, j), None)
                return f"Merged experts {i},{j} in family {m} (sim={best_sim:.3f})"
        return None

    def _recycle_experts(self):
        """Recycle dormant experts (activation frequency < tau_l)."""
        recycled = []
        family_sizes = self.model.get_family_sizes()
        for m, size in enumerate(family_sizes):
            for n in range(size):
                freq = self.activation_tracker.get_activation_frequency(m, n)
                if freq < self.recycle_threshold and size > 2:
                    self.model.recycle_expert(m, n)
                    recycled.append(f"family{m}_expert{n} (freq={freq:.4f})")
                    self.total_recycles += 1
                    # Reset tracking metadata for recycled expert
                    self.activation_tracker.activation_counts.pop((m, n), None)
                    self.activation_tracker.activation_counts[(m, n)] = deque(
                        maxlen=self.window_size
                    )
                    self.similarity_tracker.centroids.pop((m, n), None)
                    self.similarity_tracker.centroid_counts.pop((m, n), None)

        if recycled:
            self.epochs_since_structural_change = 0
            return f"Recycled {len(recycled)} experts: {', '.join(recycled)}"
        return None

    def start_phase(self, phase_idx):
        """Initialize tracking for a new incremental phase."""
        self.current_phase = phase_idx
        self.phase_epoch = 0
        self.loss_history = []
        self.stagnation_counter = 0
        self.epochs_since_structural_change = self.cooldown_epochs  # Skip cooldown at start

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
