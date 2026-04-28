"""
Logging and checkpoint utilities for HAES training.
"""

import os
import json
import logging
import torch
from datetime import datetime


def setup_logger(log_dir, name="HAES"):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def log_metrics(logger, metrics, step, prefix=""):
    """Log metrics dict to logger."""
    msg = f"[{prefix} Step {step}] " + " | ".join(
        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in metrics.items()
    )
    logger.info(msg)


def save_checkpoint(model, optimizer, epoch, phase, metrics, save_dir, filename=None):
    """Save model checkpoint with metadata."""
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        filename = f"haes_phase{phase}_epoch{epoch}.pt"

    checkpoint = {
        "epoch": epoch,
        "phase": phase,
        "model_state": model.get_state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(model, optimizer, checkpoint_path, device="cuda"):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint.get("epoch", 0), checkpoint.get("phase", 0), checkpoint.get("metrics", {})


def save_results_csv(results, save_path):
    """Save evaluation results to CSV with SHA256 verification."""
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

    # Compute SHA256 hash for reproducibility verification
    import hashlib
    with open(save_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    hash_path = save_path + ".sha256"
    with open(hash_path, "w") as f:
        f.write(f"{file_hash}  {os.path.basename(save_path)}\n")

    return file_hash
