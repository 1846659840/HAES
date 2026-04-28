"""
HAES Testing and Evaluation Script.
Loads a trained checkpoint and runs comprehensive evaluation.

Usage:
    python main_test.py --checkpoint output/checkpoints/haes_phase5_best.pt --dataset xd_violence
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.haes import HAES
from data.dataset import collate_variable_length
from data.incremental_split import IncrementalDataSplitter
from training.evaluator import Evaluator
from utils.logging import setup_logger, save_results_csv


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="HAES Model Testing")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="xd_violence",
                        choices=["xd_violence", "ucf_crime", "shanghaitech"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--feature_dir", type=str, default="./data/features")
    parser.add_argument("--output_dir", type=str, default="./output/test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference speed benchmark")
    parser.add_argument("--drift_test", action="store_true",
                        help="Run scene drift robustness test")
    parser.add_argument("--noise_test", action="store_true",
                        help="Run noise robustness test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, "HAES_test")

    # Load config and protocol
    config = load_config(args.config)
    with open("configs/incremental_protocol.yaml", "r") as f:
        protocol = yaml.safe_load(f)

    # Build model config with both nested sections and flattened keys
    full_config = {}
    for section in ["feature_extraction", "hmoe", "incremental", "elm", "training", "evaluation"]:
        if section in config:
            full_config[section] = config[section]
            full_config.update(config[section])
    # Map keys for HAES direct instantiation
    full_config["input_dim"] = config.get("feature_extraction", {}).get("feature_dim", 512)
    full_config["top_k_segments"] = config.get("training", {}).get("top_k_segments", 3)
    # Dataset-appropriate primary metric
    eval_cfg = config.get("evaluation", {})
    if args.dataset == "ucf_crime":
        full_config["primary_metric"] = "AUC"
    else:
        full_config["primary_metric"] = eval_cfg.get("xd_violence_metric", "AP")

    full_config["num_classes"] = protocol[args.dataset]["num_categories"]
    full_config["total_phases"] = protocol[args.dataset]["num_phases"]

    # Initialize model
    model = HAES(full_config)
    model.to(args.device)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state(checkpoint["model_state"])
    model.eval()

    logger.info(f"Checkpoint: Phase {checkpoint.get('phase', '?')}, Epoch {checkpoint.get('epoch', '?')}")
    logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")

    # Build test loaders
    data_dir = os.path.join(args.data_dir, args.dataset.upper().replace("_", "-")
                           ).replace("XD-VIOLENCE", "XD-Violence").replace("UCF-CRIME", "UCF-Crime")
    splitter = IncrementalDataSplitter(
        dataset_name=args.dataset,
        data_dir=data_dir,
        protocol_config=protocol,
    )
    phase_data = splitter.get_split()

    # Build test datasets
    from main_train import _create_dataset
    batch_size = config["training"]["batch_size"]

    test_loaders = []
    phase_configs = []
    for phase_idx in sorted(phase_data.keys()):
        data = phase_data[phase_idx]
        test_dataset = _create_dataset(
            data["test_videos"], data["test_labels"], args.feature_dir
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_variable_length,
            pin_memory=True,
        )
        test_loaders.append(test_loader)
        phase_configs.append((data["seen_categories"], data["categories"]))

    # Initialize evaluator
    evaluator = Evaluator(model, device=args.device)

    # 1. Evaluate overall performance
    logger.info("=" * 60)
    logger.info("Overall Performance Evaluation")
    logger.info("=" * 60)
    final_results = evaluator.evaluate(
        test_loaders[-1],
        len(phase_configs[-1][0]),
        args.dataset,
    )
    logger.info(f"Mean AP: {final_results['mean_ap']:.4f}")
    logger.info(f"Mean AUC: {final_results['mean_auc']:.4f}")

    # Per-class results
    seen_cats = phase_configs[-1][0]
    logger.info("\nPer-Class Results:")
    for i, cat in enumerate(seen_cats):
        logger.info(f"  {cat:15s}: AP={final_results['per_class_ap'][i]:.4f}, AUC={final_results['per_class_auc'][i]:.4f}")

    # 2. Incremental evaluation with BWT
    logger.info("\n" + "=" * 60)
    logger.info("Incremental Learning Evaluation")
    logger.info("=" * 60)
    inc_results = evaluator.evaluate_incremental_phases(
        test_loaders, phase_configs
    )
    logger.info(f"BWT: {inc_results['bwt']:.4f}")
    logger.info("Performance Matrix (phase_test x phase_train):")
    for row in inc_results["performance_matrix"]:
        logger.info(f"  {[f'{v:.4f}' for v in row]}")

    # 3. Inference speed benchmark
    if args.benchmark:
        logger.info("\n" + "=" * 60)
        logger.info("Inference Speed Benchmark")
        logger.info("=" * 60)
        latency = evaluator.benchmark_inference_speed(
            test_loaders[-1], num_runs=200, warmup=20
        )
        logger.info(f"P50 latency: {latency['p50_ms']:.1f} ms")
        logger.info(f"P95 latency: {latency['p95_ms']:.1f} ms")
        logger.info(f"P99 latency: {latency['p99_ms']:.1f} ms")
        logger.info(f"Mean latency: {latency['mean_ms']:.1f} ms")
        logger.info(f"FPS: {latency['fps']:.1f}")
        logger.info(f"P99/Median ratio: {latency['p99_median_ratio']:.2f}x")

        # Component breakdown
        comp_lat = evaluator.evaluate_per_component_latency(test_loaders[-1])
        logger.info(f"Routing: {comp_lat['routing_ms']:.1f} ms")
        logger.info(f"Expert Forward: {comp_lat['expert_forward_ms']:.1f} ms")
        logger.info(f"Aggregation: {comp_lat['aggregation_ms']:.1f} ms")
        logger.info(f"Total: {comp_lat['total_ms']:.1f} ms")

    # 4. Scene drift robustness test
    if args.drift_test:
        logger.info("\n" + "=" * 60)
        logger.info("Scene Drift Robustness Evaluation")
        logger.info("=" * 60)
        drift_results = evaluator.evaluate_scene_drift_robustness(
            test_loaders[-1],
            len(phase_configs[-1][0]),
        )
        for drift_type, metrics in drift_results.items():
            logger.info(f"  {drift_type:15s}: AP={metrics.get('mean_ap', 0):.4f}, AUC={metrics.get('mean_auc', 0):.4f}")

    # 5. Noise robustness test
    if args.noise_test:
        logger.info("\n" + "=" * 60)
        logger.info("Noise Robustness Evaluation")
        logger.info("=" * 60)
        noise_results = evaluator.evaluate_noise_robustness(
            test_loaders[-1],
            len(phase_configs[-1][0]),
            noise_ratio=0.15,
        )
        logger.info(f"Noise mean entropy: {noise_results['noise_mean_entropy']:.4f}")
        logger.info(f"Hard normal mean entropy: {noise_results['hard_normal_mean_entropy']:.4f}")
        logger.info(f"Noise rejection rate: {noise_results['noise_rejection_rate']:.4f}")
        logger.info(f"Hard normal retention: {noise_results['hard_normal_retention_rate']:.4f}")
        logger.info(f"Entropy AUC: {noise_results['entropy_auc']:.4f}")

    # Save results
    results_data = [{
        "metric": "mean_ap",
        "value": final_results["mean_ap"],
    }, {
        "metric": "mean_auc",
        "value": final_results["mean_auc"],
    }, {
        "metric": "bwt",
        "value": inc_results["bwt"],
    }]
    if args.benchmark:
        results_data.append({"metric": "fps", "value": latency["fps"]})
        results_data.append({"metric": "p99_ms", "value": latency["p99_ms"]})

    results_path = os.path.join(args.output_dir, f"test_results_{args.dataset}.csv")
    file_hash = save_results_csv(results_data, results_path)
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"SHA256: {file_hash}")


if __name__ == "__main__":
    main()
