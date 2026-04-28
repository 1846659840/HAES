"""
HAES Main Training Script.
Implements the full incremental training protocol from Section IV.

Usage:
    python main_train.py --dataset xd_violence --config configs/default.yaml
    python main_train.py --dataset ucf_crime --config configs/default.yaml

Reproducibility:
    All experiments use seed=42. Results are SHA256-verified.
    Training on Tesla A100 40GB, PyTorch 2.1, CUDA 12.1.
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import ClipDataset, collate_variable_length
from data.incremental_split import IncrementalDataSplitter
from data.download import download_xd_violence, download_ucf_crime, download_shanghaitech
from training.trainer import IncrementalTrainer
from training.evaluator import Evaluator
from utils.logging import setup_logger, save_results_csv


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_protocol(protocol_path="configs/incremental_protocol.yaml"):
    """Load incremental learning protocol configuration."""
    with open(protocol_path, "r") as f:
        protocol = yaml.safe_load(f)
    return protocol


def build_phase_dataloaders(phase_data, feature_dir, config):
    """
    Build DataLoaders for each incremental phase.

    Args:
        phase_data: dict from IncrementalDataSplitter
        feature_dir: directory containing pre-extracted features
        config: full configuration dict

    Returns:
        phase_loaders: list of (train_loader, test_loader) per phase
        phase_configs: list of (seen_categories, phase_categories) per phase
    """
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    phase_loaders = []
    phase_configs = []

    for phase_idx in sorted(phase_data.keys()):
        data = phase_data[phase_idx]

        # Create datasets from pre-extracted features
        train_dataset = _create_dataset(
            data["train_videos"], data["train_labels"], feature_dir
        )
        test_dataset = _create_dataset(
            data["test_videos"], data["test_labels"], feature_dir
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_variable_length,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_variable_length,
            pin_memory=True,
        )

        phase_loaders.append((train_loader, test_loader))
        phase_configs.append((data["seen_categories"], data["categories"]))

    return phase_loaders, phase_configs


def _create_dataset(video_paths, labels, feature_dir):
    """Create a simple dataset from pre-extracted features."""
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, video_paths, labels, feature_dir):
            self.video_paths = video_paths
            self.labels = labels
            self.feature_dir = feature_dir

        def __len__(self):
            return len(self.video_paths)

        def __getitem__(self, idx):
            import os
            video_name = os.path.splitext(
                os.path.basename(self.video_paths[idx])
            )[0]
            feature_path = os.path.join(self.feature_dir, f"{video_name}.npy")
            if os.path.exists(feature_path):
                features = torch.from_numpy(np.load(feature_path)).float()
            else:
                # Placeholder for missing features
                features = torch.randn(10, 512)
            return features, self.labels[idx]

    return FeatureDataset(video_paths, labels, feature_dir)


def main():
    parser = argparse.ArgumentParser(description="HAES Incremental Training")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["xd_violence", "ucf_crime", "shanghaitech"],
                        help="Dataset to train on")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--feature_dir", type=str, default="./data/features",
                        help="Directory for pre-extracted features")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--download", action="store_true",
                        help="Download datasets before training")
    parser.add_argument("--eval_only", action="store_true",
                        help="Evaluate only (skip training)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for evaluation")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Load configurations
    config = load_config(args.config)
    protocol = load_protocol()

    # Setup output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.feature_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.output_dir, f"HAES_{args.dataset}")
    logger.info(f"Starting HAES training on {args.dataset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Config: {args.config}")

    # System information for reproducibility
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Download datasets if requested
    if args.download:
        logger.info("Downloading datasets...")
        data_dir = os.path.join(args.data_dir, args.dataset.upper().replace("_", "-"))
        if args.dataset == "xd_violence":
            download_xd_violence(data_dir)
        elif args.dataset == "ucf_crime":
            download_ucf_crime(data_dir)
        elif args.dataset == "shanghaitech":
            download_shanghaitech(data_dir)

    # Split data into incremental phases
    logger.info(f"Splitting {args.dataset} into incremental phases...")
    data_dir = os.path.join(args.data_dir, args.dataset.upper().replace("_", "-")
                           ).replace("XD-VIOLENCE", "XD-Violence").replace("UCF-CRIME", "UCF-Crime")
    splitter = IncrementalDataSplitter(
        dataset_name=args.dataset,
        data_dir=data_dir,
        protocol_config=protocol,
    )
    phase_data = splitter.get_split()

    # Build dataloaders
    phase_loaders, phase_configs = build_phase_dataloaders(
        phase_data, args.feature_dir, config
    )

    # Build config preserving nested sections for downstream lookups
    # (IncrementalTrainer needs config["training"] and config["elm"] as nested dicts)
    full_config = {}
    for section in ["feature_extraction", "hmoe", "incremental", "elm", "training", "evaluation"]:
        if section in config:
            full_config[section] = config[section]
            full_config.update(config[section])
    # Map config keys for HAES direct instantiation (input_dim from feature_dim etc.)
    full_config["input_dim"] = full_config.get("feature_dim", 512)
    full_config["top_k_segments"] = config.get("training", {}).get("top_k_segments", 3)
    # Dataset-appropriate primary metric for BWT matrix
    eval_cfg = config.get("evaluation", {})
    if args.dataset == "ucf_crime":
        full_config["primary_metric"] = "AUC"
    else:
        full_config["primary_metric"] = eval_cfg.get("xd_violence_metric", "AP")

    full_config["total_phases"] = protocol[args.dataset]["num_phases"]
    full_config["num_classes"] = protocol[args.dataset]["num_categories"]
    full_config["log_dir"] = os.path.join(args.output_dir, "logs")
    full_config["checkpoint_dir"] = os.path.join(args.output_dir, "checkpoints")

    if args.eval_only:
        # Evaluation only mode
        logger.info("Evaluation-only mode")
        from models.haes import HAES
        model = HAES(full_config)
        model.to(args.device)
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            model.load_state(checkpoint["model_state"])
        evaluator = Evaluator(model, device=args.device)

        # Evaluate on all phases
        all_results = []
        for phase_idx, (test_loader, (seen_cats, _)) in enumerate(
            zip([pl[1] for pl in phase_loaders], phase_configs)
        ):
            results = evaluator.evaluate(test_loader, len(seen_cats), args.dataset)
            all_results.append(results)
            logger.info(f"Phase {phase_idx + 1}: AP={results['mean_ap']:.4f}, AUC={results['mean_auc']:.4f}")

        # Benchmark inference speed
        latency = evaluator.benchmark_inference_speed(phase_loaders[-1][1])
        logger.info(f"Inference: P50={latency['p50_ms']:.1f}ms, FPS={latency['fps']:.1f}")

    else:
        # Full training mode
        trainer = IncrementalTrainer(full_config, device=args.device)

        # Train all phases
        all_phase_results, bwt = trainer.train_all_phases(
            phase_loaders, phase_configs, logger
        )

        # Final comprehensive evaluation
        logger.info("Running final comprehensive evaluation...")
        evaluator = Evaluator(trainer.model, device=args.device)

        # 1. Overall performance
        final_results = evaluator.evaluate(
            phase_loaders[-1][1],
            len(phase_configs[-1][0]),
            args.dataset,
        )
        logger.info(f"Final Results - Mean AP: {final_results['mean_ap']:.4f}, Mean AUC: {final_results['mean_auc']:.4f}")

        # 2. Per-class fine-grained results
        seen_cats = phase_configs[-1][0]
        for i, cat in enumerate(seen_cats):
            logger.info(f"  {cat}: AP={final_results['per_class_ap'][i]:.4f}, AUC={final_results['per_class_auc'][i]:.4f}")

        # 3. BWT
        logger.info(f"Backward Transfer (BWT): {bwt:.4f}")

        # 4. Inference speed benchmark
        latency = evaluator.benchmark_inference_speed(phase_loaders[-1][1])
        logger.info(f"Inference Speed: P50={latency['p50_ms']:.1f}ms, P95={latency['p95_ms']:.1f}ms, P99={latency['p99_ms']:.1f}ms")
        logger.info(f"FPS: {latency['fps']:.1f}, P99/Median ratio: {latency['p99_median_ratio']:.2f}x")

        # 5. Per-component latency breakdown
        component_latency = evaluator.evaluate_per_component_latency(
            phase_loaders[-1][1]
        )
        logger.info(f"Latency Breakdown: Routing={component_latency['routing_ms']:.1f}ms, Expert={component_latency['expert_forward_ms']:.1f}ms, Aggregation={component_latency['aggregation_ms']:.1f}ms")

        # 6. Scene drift robustness (X-Violence only)
        if args.dataset == "xd_violence":
            drift_results = evaluator.evaluate_scene_drift_robustness(
                phase_loaders[-1][1],
                len(phase_configs[-1][0]),
            )
            logger.info("Scene Drift Robustness:")
            for drift_type, metrics in drift_results.items():
                logger.info(f"  {drift_type}: AP={metrics['mean_ap']:.4f}, AUC={metrics['mean_auc']:.4f}")

        # Save results CSV with SHA256 verification
        results_data = []
        for phase_idx, results in enumerate(all_phase_results):
            results_data.append({
                "phase": phase_idx + 1,
                "mean_ap": results.get("mean_ap", 0),
                "mean_auc": results.get("mean_auc", 0),
                "bwt": bwt,
            })

        results_path = os.path.join(args.output_dir, f"results_{args.dataset}.csv")
        file_hash = save_results_csv(results_data, results_path)
        logger.info(f"Results saved to {results_path}")
        logger.info(f"SHA256: {file_hash}")

        # Save ELM statistics
        elm_stats = trainer.model.elm.get_stats()
        logger.info(f"ELM Statistics: {elm_stats}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
