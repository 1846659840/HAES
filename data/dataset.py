"""
Video dataset and clip extraction for HAES.

Implements the feature extraction pipeline from Section III-A:
- Raw video -> clip segmentation (sliding window)
- Per-channel normalization
- R3D-18 feature extraction (frozen, Kinetics-400 pretrained)
- Temporal feature sequence generation
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.video import r3d_18, R3D_18_Weights


class VideoClipExtractor:
    """
    Extracts fixed-length clips from videos using a sliding window.
    Section III-A: T=16 frames, stride delta_T=16 frames.
    """

    def __init__(self, clip_length=16, clip_stride=16, input_size=(112, 112)):
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.input_size = input_size
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989]),
        ])

    def extract_clips(self, video_path):
        """Extract clips from a video file. Returns list of clip tensors."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return []

        clips = []
        for start in range(0, len(frames) - self.clip_length + 1, self.clip_stride):
            clip_frames = frames[start:start + self.clip_length]
            clip_tensor = torch.stack([self.transform(f) for f in clip_frames])
            # R3D-18 expects [C, T, H, W]; stack produces [T, C, H, W]
            clip_tensor = clip_tensor.permute(1, 0, 2, 3)
            clips.append(clip_tensor)  # [C, T, H, W]
        return clips


class FeatureExtractor:
    """
    R3D-18 feature extractor with Kinetics-400 pretrained weights.
    Frozen during all training phases (Section III-A).
    Output: 512-dim clip feature vectors.
    """

    def __init__(self, device="cuda"):
        self.device = device
        weights = R3D_18_Weights.KINETICS400_V1
        self.model = r3d_18(weights=weights)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # Remove the final classifier to get 512-dim features
        self.model.fc = torch.nn.Identity()

    @torch.no_grad()
    def extract_features(self, clips):
        """
        Extract features from clips.

        Args:
            clips: list of tensors [C, T, H, W] or stacked tensor [N, C, T, H, W]

        Returns:
            Feature tensor [N, 512] or [N, T', 512] if returning spatial features
        """
        if isinstance(clips, list):
            clips = torch.stack(clips)
        clips = clips.to(self.device)
        features = self.model(clips)
        return features.cpu()  # [N, 512]


class VideoDataset(Dataset):
    """
    Dataset for loading pre-extracted features or raw videos.

    For efficiency, features are pre-extracted and cached.
    Each video yields a sequence of clip features F in R^{T_seg x 512}.
    """

    def __init__(self, video_list, label_list, feature_dir=None,
                 clip_length=16, clip_stride=16, input_size=(112, 112),
                 device="cuda", cache_features=True):
        self.video_list = video_list
        self.label_list = label_list
        self.feature_dir = feature_dir
        self.cache_features = cache_features
        self.device = device
        self.clip_extractor = VideoClipExtractor(
            clip_length=clip_length,
            clip_stride=clip_stride,
            input_size=input_size
        )
        if cache_features and feature_dir:
            self.feature_extractor = FeatureExtractor(device=device)

    def __len__(self):
        return len(self.video_list)

    def _get_cached_path(self, video_path):
        """Get the cache path for pre-extracted features."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(self.feature_dir, f"{video_name}.npy")

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        label = self.label_list[idx]

        # Check if features are cached
        if self.cache_features and self.feature_dir:
            cache_path = self._get_cached_path(video_path)
            if os.path.exists(cache_path):
                features = torch.from_numpy(np.load(cache_path))
                return features, label

        # Extract clips and features
        clips = self.clip_extractor.extract_clips(video_path)
        if len(clips) == 0:
            return torch.zeros(1, 512), label

        if self.cache_features and self.feature_dir:
            features = self.feature_extractor.extract_features(clips)
            # Cache for future use
            os.makedirs(self.feature_dir, exist_ok=True)
            cache_path = self._get_cached_path(video_path)
            np.save(cache_path, features.numpy())
        else:
            features = torch.stack(clips)  # placeholder

        return features, label


class ClipDataset(Dataset):
    """
    Dataset that loads pre-extracted clip features.
    Each sample is F in R^{T_seg x 512} with a video-level label.
    """

    def __init__(self, features_dir, annotation_file):
        """
        Args:
            features_dir: directory containing .npy feature files
            annotation_file: CSV with columns [video_name, label]
        """
        import pandas as pd
        self.features_dir = features_dir
        self.annotations = pd.read_csv(annotation_file)
        self.video_names = self.annotations['video_name'].tolist()
        self.labels = self.annotations['label'].tolist()
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        label = self.labels[idx]
        feature_path = os.path.join(self.features_dir, f"{video_name}.npy")
        features = torch.from_numpy(np.load(feature_path)).float()
        return features, label


def collate_variable_length(batch):
    """
    Custom collate for variable-length feature sequences.
    Pads sequences to max length in batch.
    """
    features, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features])
    max_len = max(lengths)

    padded = torch.zeros(len(features), max_len, features[0].size(1))
    for i, f in enumerate(features):
        padded[i, :f.size(0), :] = f

    labels = torch.tensor(labels)
    return padded, labels, lengths
