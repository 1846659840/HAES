"""
Incremental Learning Data Splitter for HAES.

Implements the incremental protocols from Section IV-A:
- XD-Violence: 6 phases (Abuse+CarAccident -> Explosion -> Fighting -> Riot -> Shooting -> Normal)
- UCF-Crime: 13 phases (one new anomaly category per phase)
- No exemplar replay from previous phases (strict privacy constraint)
"""

import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict


class IncrementalDataSplitter:
    """
    Splits datasets into incremental phases following the paper protocol.
    Previous phase data is strictly inaccessible in subsequent phases.
    """

    def __init__(self, dataset_name, data_dir, protocol_config):
        """
        Args:
            dataset_name: "xd_violence" or "ucf_crime" or "shanghaitech"
            data_dir: root directory of the dataset
            protocol_config: dict with phase definitions from incremental_protocol.yaml
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.config = protocol_config[dataset_name]
        # ShanghaiTech has different structure (no phases/num_phases keys)
        self.phases = self.config.get("phases", [])
        self.num_phases = self.config.get("num_phases", 1)
        self.categories = self.config["categories"]

    def split_xd_violence(self, annotation_dir=None):
        """
        Split XD-Violence following the paper's 6-phase protocol.

        Phase 1: Abuse, CarAccident
        Phase 2: Explosion
        Phase 3: Fighting
        Phase 4: Riot
        Phase 5: Shooting
        Phase 6: Normal (background)

        Returns:
            phase_data: dict mapping phase_idx -> (train_videos, train_labels, test_videos, test_labels)
        """
        if annotation_dir is None:
            annotation_dir = os.path.join(self.data_dir, "annotations")

        train_ann = pd.read_csv(os.path.join(annotation_dir, "train_annotation.csv"))
        test_ann = pd.read_csv(os.path.join(annotation_dir, "test_annotation.csv"))

        # Map category names to indices
        cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}

        phase_data = {}
        all_seen_categories = set()

        for phase_idx, phase_categories in enumerate(self.phases):
            all_seen_categories.update(phase_categories)

            # Training data: only current phase categories (no replay from past)
            train_mask = train_ann['label'].isin(phase_categories)
            train_subset = train_ann[train_mask]

            train_videos = [
                os.path.join(self.data_dir, "videos", "train", row['label'], row['video_path'])
                for _, row in train_subset.iterrows()
            ]
            train_labels = [cat_to_idx[row['label']] for _, row in train_subset.iterrows()]

            # Test data: all categories seen so far (evaluating on cumulative knowledge)
            test_mask = test_ann['label'].isin(all_seen_categories)
            test_subset = test_ann[test_mask]

            test_videos = [
                os.path.join(self.data_dir, "videos", "test", row['label'], row['video_path'])
                for _, row in test_subset.iterrows()
            ]
            test_labels = [cat_to_idx[row['label']] for _, row in test_subset.iterrows()]

            phase_data[phase_idx] = {
                "train_videos": train_videos,
                "train_labels": train_labels,
                "test_videos": test_videos,
                "test_labels": test_labels,
                "categories": list(phase_categories),
                "seen_categories": list(all_seen_categories),
            }

        return phase_data

    def split_ucf_crime(self, annotation_dir=None):
        """
        Split UCF-Crime following the paper's 13-phase protocol.
        Each phase introduces one new anomaly category.

        Phase order: Abuse, Arrest, Arson, Assault, Burglary, Explosion,
                     Fighting, RoadAcc, Robbery, Shooting, Shoplifting, Stealing, Vandalism

        Returns:
            phase_data: dict mapping phase_idx -> (train_videos, train_labels, ...)
        """
        if annotation_dir is None:
            annotation_dir = os.path.join(self.data_dir, "annotations")

        train_ann = pd.read_csv(os.path.join(annotation_dir, "train_annotations.csv"))
        test_ann = pd.read_csv(os.path.join(annotation_dir, "test_annotations.csv"))

        cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}

        phase_data = {}
        all_seen_categories = set()

        for phase_idx, phase_categories in enumerate(self.phases):
            all_seen_categories.update(phase_categories)

            train_mask = train_ann['label'].isin(phase_categories)
            train_subset = train_ann[train_mask]

            train_videos = [
                os.path.join(self.data_dir, "videos", "train", row['label'], row['video_path'])
                for _, row in train_subset.iterrows()
            ]
            train_labels = [cat_to_idx[row['label']] for _, row in train_subset.iterrows()]

            test_mask = test_ann['label'].isin(all_seen_categories)
            test_subset = test_ann[test_mask]

            test_videos = [
                os.path.join(self.data_dir, "videos", "test", row['label'], row['video_path'])
                for _, row in test_subset.iterrows()
            ]
            test_labels = [cat_to_idx[row['label']] for _, row in test_subset.iterrows()]

            phase_data[phase_idx] = {
                "train_videos": train_videos,
                "train_labels": train_labels,
                "test_videos": test_videos,
                "test_labels": test_labels,
                "categories": list(phase_categories),
                "seen_categories": list(all_seen_categories),
            }

        return phase_data

    def split_shanghaitech(self):
        """
        Split ShanghaiTech for cross-scenario generalization testing.
        """
        # Training videos from training split
        train_dir = os.path.join(self.data_dir, "training", "videos")
        test_dir = os.path.join(self.data_dir, "testing", "videos")

        train_videos = [
            os.path.join(train_dir, f)
            for f in sorted(os.listdir(train_dir))
            if f.endswith(('.mp4', '.avi'))
        ]

        test_videos = [
            os.path.join(test_dir, f)
            for f in sorted(os.listdir(test_dir))
            if f.endswith(('.mp4', '.avi'))
        ]

        return {
            "train_videos": train_videos,
            "test_videos": test_videos,
        }

    def get_split(self, annotation_dir=None):
        """Get the dataset split based on dataset type."""
        if self.dataset_name == "xd_violence":
            return self.split_xd_violence(annotation_dir)
        elif self.dataset_name == "ucf_crime":
            return self.split_ucf_crime(annotation_dir)
        elif self.dataset_name == "shanghaitech":
            return self.split_shanghaitech()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


class SyntheticIncrementalStream:
    """
    Creates synthetic incremental streams for robustness testing
    (Appendix M: scene drift, periodic distribution drift, etc.).
    """

    def __init__(self, base_data, drift_type="illumination"):
        self.base_data = base_data
        self.drift_type = drift_type

    def apply_illumination_drift(self, features, gamma=0.8):
        """Simulate day-to-night illumination transition."""
        features = features.clone()
        noise = torch.randn_like(features) * 0.1
        return features * gamma + noise

    def apply_fov_shift(self, features, translation=0.06):
        """Simulate camera bump / field-of-view shift."""
        features = features.clone()
        noise = torch.randn_like(features[:, :, :64]) * translation
        features[:, :, :64] += noise
        return features

    def apply_codec_compression(self, features, quality=20):
        """Simulate bandwidth-driven re-encoding artifacts."""
        features = features.clone()
        noise = torch.randn_like(features) * (1.0 / quality)
        return features + noise

    def apply_low_light(self, features, sigma=0.1):
        """Simulate nighttime/infrared sensor noise (Appendix C)."""
        noise = torch.randn_like(features) * sigma
        return features + noise

    def apply_low_resolution(self, features):
        """Simulate resolution degradation (downsample + upsample)."""
        # Simulate through dimensionality reduction/recovery in feature space
        features = features.clone()
        d = features.size(-1)
        U, S, Vh = torch.linalg.svd(features.reshape(-1, d).float(), full_matrices=False)
        S[d // 2:] *= 0.3  # Attenuate high-frequency components
        return (U @ torch.diag(S) @ Vh).reshape_as(features)

    def apply_infrared_shift(self, features, channel_shift=0.15):
        """Simulate visible-to-infrared spectral shift."""
        features = features.clone()
        noise = torch.ones_like(features) * channel_shift
        return features + noise

    def generate_drifted_batch(self, features, drift_type=None):
        """Apply specified distribution drift to features."""
        if drift_type is None:
            drift_type = self.drift_type

        drift_fn = {
            "illumination": self.apply_illumination_drift,
            "fov_shift": self.apply_fov_shift,
            "codec_compression": self.apply_codec_compression,
            "low_light": self.apply_low_light,
            "low_resolution": self.apply_low_resolution,
            "infrared": self.apply_infrared_shift,
        }

        if drift_type in drift_fn:
            return drift_fn[drift_type](features)
        return features
