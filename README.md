# HAES: Hierarchical Adaptive Expert System

**A Scalable Expert-based System for Violence Detection with Dynamic Structure Management**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**Full PyTorch implementation** | Training scripts | Per-phase hyperparameter YAMLs | Pre-computed feature caches

---

## Overview

HAES is a Hierarchical Adaptive Expert System for weakly-supervised incremental violence detection in surveillance videos. The system combines a Hierarchical Mixture of Experts (HMoE) architecture with Expert Lifecycle Management (ELM) to enable continuous learning of new violence categories while preserving previously acquired knowledge.

### Key Features

- **Two-Level Sparse Routing**: Hierarchical gating with cluster-level (Top-k1) and intra-cluster (Top-k2) expert selection
- **Incremental Learning**: Output consistency (KD), feature preservation (MSE), routing distillation (R-KL), and EWC constraints
- **Expert Lifecycle Management**: Dynamic expert addition, merging, and recycling based on activation frequency and feature similarity
- **Noise Suppression**: Entropy-weighted distillation, temporal consistency filtering, multi-round memory verification
- **Real-Time Inference**: Bounded O(k1*k2) routing ensures constant inference cost regardless of total expert count

### Performance

| Dataset | Metric | Score |
|---------|--------|-------|
| XD-Violence | AP | 89.68% |
| UCF-Crime | AUC | 88.56% |
| Backward Transfer | BWT | -2.0% |

---

## Repository Structure

```
HAES/
├── configs/
│   ├── default.yaml              # Default hyperparameters (Section IV-B)
│   └── incremental_protocol.yaml # Phase definitions for each dataset
├── data/
│   ├── __init__.py
│   ├── download.py               # Dataset download utilities
│   ├── dataset.py                # VideoDataset, ClipDataset, FeatureExtractor
│   └── incremental_split.py      # Incremental phase splitting (Section IV-A)
├── models/
│   ├── __init__.py
│   ├── haes.py                   # Full HAES model (Eq. 22)
│   ├── hmoe.py                   # Hierarchical Mixture of Experts (Section III-B)
│   ├── gating.py                 # Two-level gating networks (Eq. 5-6)
│   ├── experts.py                # Transformer expert blocks (Eq. 7)
│   ├── constraints.py            # KD, MSE, R-KL, EWC, Temporal Consistency
│   └── elm.py                    # Expert Lifecycle Management (Section III-E)
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Incremental training loop
│   └── evaluator.py              # Comprehensive evaluation suite
├── utils/
│   ├── __init__.py
│   ├── metrics.py                # AP, AUC, BWT computation (Eq. 25-27)
│   └── logging.py                # Logging, checkpointing, SHA256 verification
├── main_train.py                 # Main training entry point
├── main_test.py                  # Testing and evaluation entry point
├── environment.yaml              # Conda environment specification
├── Dockerfile                    # Docker image for reproducibility
├── Makefile                      # Reproducibility targets
├── requirements.txt              # Pip dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## System Requirements

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Tesla A100 40GB |
| NVIDIA Driver | 535.104.05 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Python | 3.10.13 |
| OS | Ubuntu 22.04 LTS |

**Minimum Requirements**: NVIDIA GPU with 16GB+ VRAM (RTX 3090 or higher recommended for full training).

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/1846659840/HAES.git
cd HAES
```

### 2. Setup Environment

**Option A: Conda (Recommended)**

```bash
conda env create -f environment.yaml
conda activate haes
```

**Option B: Docker**

```bash
# Docker image SHA256: will be generated at build time
docker build -t haes:latest .
docker run --gpus all -v $(pwd)/data:/workspace/HAES/data haes:latest
```

**Option C: Pip**

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

```bash
# Download instructions provided; datasets require manual download from official sources
python -c "from data.download import download_xd_violence; download_xd_violence()"
python -c "from data.download import download_ucf_crime; download_ucf_crime()"
python -c "from data.download import download_shanghaitech; download_shanghaitech()"
```

**Dataset Sources:**
- **XD-Violence**: [GitHub Repository](https://github.com/YoonBoWon/XD-Violence) - 4,754 videos, 217 hours, 6 violence categories
- **UCF-Crime**: [Dataset Page](https://webpages.charlotte.edu/cchen62/dataset.html) - 1,900 real-world surveillance videos, 13 anomaly categories
- **ShanghaiTech Campus**: [GitHub Repository](https://github.com/StevenLiuWen/anoPred_cvpr2018) - 13 scenes with complex anomaly patterns

### 4. Reproduce Results

```bash
# Reproduce all experiments (ensures SHA256-verified results CSV)
make reproduce_all

# Or run individual datasets
make reproduce_xd       # XD-Violence 6 phases
make reproduce_ucf      # UCF-Crime 13 phases
make reproduce_shanghaitech  # ShanghaiTech cross-scenario
```

This runs all incremental phases under the default seed (42) and emits a SHA256-verified results CSV at `output/<dataset>/results_<dataset>.csv`.

### 5. Custom Training

```bash
# XD-Violence with custom config
python main_train.py --dataset xd_violence --config configs/default.yaml --seed 42

# UCF-Crime with custom data directory
python main_train.py --dataset ucf_crime --data_dir /path/to/data --output_dir ./my_output

# Evaluation only
python main_test.py --checkpoint output/checkpoints/haes_phase5_best.pt --dataset xd_violence --benchmark --drift_test
```

---

## Configuration

All hyperparameters are specified in `configs/default.yaml` with per-phase values matching Section IV-B:

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| Clip Length T | 16 | Sliding window frames |
| Clip Stride | 16 | Frame step between clips |
| Latent Dim D | 512 | HMoE latent dimension |
| Num Families M | 4 | Expert family count |
| Experts per Family | 3 | Initial experts per family |
| Top-k1 | 2 | Family-level sparse selection |
| Top-k2 | 2 | Expert-level sparse selection |
| Temperature τ | 4.0 | Distillation temperature |
| λ_KD | 1.0 | Output distillation weight |
| λ_MSE | 1.0 | Feature consistency weight |
| λ_R | 1.0 | Routing preservation weight |
| λ_EWC | 100.0 | EWC regularization weight |
| Learning Rate | 5e-4 | Adam optimizer |
| Batch Size | 64 | Training batch size |
| Warmup Epochs | 3 | Uniform distillation before activation |

### Incremental Protocol

**XD-Violence (6 phases)**:
| Phase | New Categories |
|-------|---------------|
| 1 | Abuse, CarAccident |
| 2 | Explosion |
| 3 | Fighting |
| 4 | Riot |
| 5 | Shooting |
| 6 | Normal (background) |

**UCF-Crime (13 phases)**:
| Phase | Category | Phase | Category |
|-------|----------|-------|----------|
| 1 | Abuse | 8 | RoadAcc |
| 2 | Arrest | 9 | Robbery |
| 3 | Arson | 10 | Shooting |
| 4 | Assault | 11 | Shoplifting |
| 5 | Burglary | 12 | Stealing |
| 6 | Explosion | 13 | Vandalism |
| 7 | Fighting | | |

---

## Model Architecture

### Hierarchical Mixture of Experts (HMoE) - Section III-B

```
Input: F_seq [B, T_seg, 512]
  |
  v
Feature Encoding + Positional Embedding (Eq. 2-4)
  |
  v
Stage-1 Family Gate (Top-k1) --> Activated Families M
  |
  v
Stage-2 Expert Gate (Top-k2) --> Activated Experts E_m per family
  |
  v
Transformer Expert Forward Pass (Eq. 7)
  |
  v
Intra-family Fusion (Eq. 8) --> H_m
  |
  v
Cross-family Fusion (Eq. 9) --> H_fused [B, T_seg, D]
  |
  v
Anomaly Scoring Head (Eq. 20) --> Segment/Video scores
```

### Constraints (Section III-C)

1. **Output KD** (Eq. 12): Temperature-scaled truncated KL on Top-k class support
2. **Feature MSE** (Eq. 13): L2 consistency between Student and Teacher fused representations
3. **Routing KL** (Eq. 17): Family-level + Expert-level routing distribution preservation
4. **EWC** (Eq. 19): Fisher-weighted parameter anchoring to Phase-1 values

### ELM Operations (Section III-E)

- **Addition** (Eq. 23): Insert expert when family activation > τ_u^add and loss stagnates
- **Merging** (Eq. 24): Weighted-average merge for experts with cos_sim > τ_s
- **Recycling**: Reset experts with activation frequency < τ_l

---

## Evaluation Metrics

| Metric | Formula | Dataset |
|--------|---------|---------|
| AP | Eq. 25: ∫ p(r) dr | XD-Violence |
| AUC | Eq. 26: ∫ TPR(f) df | UCF-Crime |
| BWT | Eq. 27: mean(R_T,i - R_i,i) | Both |

---

## Reproducibility

### Pinned Environment

- **Docker Image SHA256**: Generated at build time via `docker build`
- **Conda Environment**: `environment.yaml` with exact package versions
- **GPU**: Tesla A100 40GB
- **CUDA**: 12.1
- **Driver**: NVIDIA 535.104.05
- **PyTorch**: 2.1.2

### Deterministic Training

All experiments use:
- `seed=42` (Python, NumPy, PyTorch, CUDA)
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `torch.backends.cudnn.deterministic=True`
- `torch.backends.cudnn.benchmark=False`

### Results Verification

```bash
# SHA256 verification of results CSV
sha256sum output/xd_violence/results_xd_violence.csv
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{haes2025,
  title={A Scalable Expert-based System for Violence Detection
         with Dynamic Structure Management},
  author={Anonymous Authors},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  year={2025},
  note={Under Review}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 HAES Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

- **Repository**: [https://github.com/1846659840/HAES](https://github.com/1846659840/HAES)
- **Issues**: Please open a GitHub issue for bug reports or feature requests.
