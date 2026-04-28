"""
Dataset download utilities for XD-Violence, UCF-Crime, and ShanghaiTech.

XD-Violence: https://github.com/YoonBoWon/XD-Violence
UCF-Crime: https://webpages.charlotte.edu/cchen62/dataset.html
ShanghaiTech: https://github.com/StevenLiuWen/anoPred_cvpr2018
"""

import os
import urllib.request
import zipfile
import tarfile
import shutil
from tqdm import tqdm


class DownloadProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        downloaded = block_num * block_size
        self.pbar.update(min(block_size, total_size - self.pbar.n))
        if downloaded >= total_size:
            self.pbar.close()
            self.pbar = None


def download_file(url, dest_path):
    """Download a file from URL to destination with progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[SKIP] {dest_path} already exists.")
        return
    print(f"Downloading {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path, DownloadProgressBar())


def download_xd_violence(data_dir="./data/XD-Violence"):
    """
    Download XD-Violence dataset.

    XD-Violence contains 4,754 videos (217 hours) across 6 violence categories:
    Abuse, CarAccident, Explosion, Fighting, Riot, Shooting + Normal.
    Training: 3,954 videos; Test: 800 videos (500 violent, 300 non-violent).

    Note: The dataset requires manual download from the official source.
    This function provides the structure and instructions.
    """
    os.makedirs(data_dir, exist_ok=True)

    readme_path = os.path.join(data_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("""# XD-Violence Dataset

Please download the XD-Violence dataset from:
https://github.com/YoonBoWon/XD-Violence

## Directory Structure Expected:
```
XD-Violence/
├── videos/
│   ├── train/
│   │   ├── Abuse/
│   │   ├── CarAccident/
│   │   ├── Explosion/
│   │   ├── Fighting/
│   │   ├── Riot/
│   │   ├── Shooting/
│   │   └── Normal/
│   └── test/
│       ├── Abuse/
│       ├── ... (same structure)
│       └── Normal/
├── annotations/
│   ├── train_annotation.csv
│   └── test_annotation.csv
└── README.md
```

## Annotation Format:
Each CSV should have columns: video_path, label, start_frame, end_frame
- video_path: relative path to video file
- label: category name (Abuse, CarAccident, Explosion, Fighting, Riot, Shooting, Normal)
- start_frame, end_frame: temporal annotation boundaries (for normal: -1, -1)
""")
    return data_dir


def download_ucf_crime(data_dir="./data/UCF-Crime"):
    """
    Download UCF-Crime dataset.

    UCF-Crime contains 1,900 real-world surveillance videos.
    Training: 1,610 videos; Test: 290 videos.
    13 anomaly categories + Normal.
    """
    os.makedirs(data_dir, exist_ok=True)

    readme_path = os.path.join(data_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("""# UCF-Crime Dataset

Please download the UCF-Crime dataset from:
https://webpages.charlotte.edu/cchen62/dataset.html

## Directory Structure Expected:
```
UCF-Crime/
├── videos/
│   ├── train/
│   │   ├── Abuse/
│   │   ├── Arrest/
│   │   ├── Arson/
│   │   ├── Assault/
│   │   ├── Burglary/
│   │   ├── Explosion/
│   │   ├── Fighting/
│   │   ├── RoadAcc/
│   │   ├── Robbery/
│   │   ├── Shooting/
│   │   ├── Shoplifting/
│   │   ├── Stealing/
│   │   ├── Vandalism/
│   │   └── Normal/
│   └── test/
│       ├── ... (same structure)
│       └── Normal/
└── annotations/
    ├── train_annotations.csv
    └── test_annotations.csv
```
""")
    return data_dir


def download_shanghaitech(data_dir="./data/ShanghaiTech"):
    """
    Download ShanghaiTech Campus dataset.

    ShanghaiTech contains 13 scenes with complex anomaly patterns.
    """
    os.makedirs(data_dir, exist_ok=True)

    readme_path = os.path.join(data_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("""# ShanghaiTech Campus Dataset

Please download from:
https://github.com/StevenLiuWen/anoPred_cvpr2018

## Directory Structure Expected:
```
ShanghaiTech/
├── training/
│   ├── videos/
│   └── frames/
├── testing/
│   ├── videos/
│   └── frames/
└── annotations/
    └── test_frame_mask/
```
""")
    return data_dir
