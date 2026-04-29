"""
Evaluation metrics for weakly-supervised video anomaly detection.

Metrics implemented per Section IV-A:
- AP (Average Precision) for XD-Violence (Eq. 25):
  AP = integral_{0}^{1} p(r) dr
  p_interp(r_{k+1}) = max_{r_tilde >= r_{k+1}} p(r_tilde)

- AUC (Area Under ROC Curve) for UCF-Crime (Eq. 26):
  AUC = integral_{0}^{1} TPR(f) df
  where TPR = TP/(TP+FN), FPR = FP/(FP+TN)
  Discrete computation via trapezoidal rule:
  AUC = sum_i (1/2) * (FPR_i - FPR_{i-1}) * (TPR_i + TPR_{i-1})

- BWT (Backward Transfer) for catastrophic forgetting (Eq. 27):
  BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
  where R_{T,i} = performance on phase i after all T phases,
  R_{i,i} = performance on phase i right after training it.
  Negative BWT = forgetting; closer to zero = better retention.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def compute_pr_curve(scores, labels, num_points=100):
    """
    Compute precision-recall curve.
    Section IV-A: AP is area under PR curve.

    Args:
        scores: anomaly scores [N]
        labels: ground truth labels (0=normal, 1=anomaly) [N]
        num_points: number of interpolation points

    Returns:
        precision: [num_points]
        recall: [num_points]
    """
    # Sort by score descending
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    total_pos = np.sum(labels == 1)
    total_neg = np.sum(labels == 0)

    if total_pos == 0:
        return np.zeros(num_points), np.zeros(num_points)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / total_pos

    # Interpolated precision (Eq. 25 discrete form)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Resample to fixed number of points
    if len(recall) > 1:
        recall_grid = np.linspace(0, 1, num_points)
        precision_grid = np.interp(recall_grid, recall, precision)
        return precision_grid, recall_grid

    return precision, recall


def compute_ap(scores, labels):
    """
    Compute Interpolated Average Precision (Eq. 25).
    XD-Violence evaluation metric.

    AP = integral_{0}^{1} p(r) dr
    p_interp(r_{k+1}) = max_{r_tilde >= r_{k+1}} p(r_tilde)

    Uses interpolated precision per the paper specification.
    This differs from sklearn's non-interpolated average_precision_score.

    Args:
        scores: anomaly scores [N]
        labels: ground truth labels [N]

    Returns:
        ap: interpolated average precision value
    """
    precision, recall = compute_pr_curve(scores, labels)
    # AP = integral of interpolated precision over recall
    if len(recall) > 1:
        trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        return float(trapz(precision, recall))
    return 0.0


def compute_auc(scores, labels):
    """
    Compute Area Under ROC Curve (Eq. 26).
    UCF-Crime evaluation metric.

    AUC = integral_{0}^{1} TPR(f) df
    Discrete: sum_i (1/2) * (FPR_i - FPR_{i-1}) * (TPR_i + TPR_{i-1})

    Args:
        scores: anomaly scores [N]
        labels: ground truth labels [N]

    Returns:
        auc: area under ROC curve
    """
    return roc_auc_score(labels, scores)


def compute_auc_trapezoidal(scores, labels):
    """
    Compute AUC using the trapezoidal rule as specified in the paper.
    AUC = sum_i (1/2) * (FPR_i - FPR_{i-1}) * (TPR_i + TPR_{i-1})
    """
    fpr, tpr, _ = roc_curve(labels, scores)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += 0.5 * (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1])

    return auc


def compute_bwt(phase_performances):
    """
    Compute Backward Transfer (BWT) metric (Eq. 27).

    BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})

    where:
    - R_{T,i}: performance on phase i after completing all T phases
    - R_{i,i}: performance on phase i immediately after its training concludes
    - T: total number of phases

    Args:
        phase_performances: 2D list or array [T x T]
            phase_performances[t][i] = performance on phase i after phase t

    Returns:
        bwt: backward transfer value (negative = forgetting)
    """
    T = len(phase_performances)
    if T <= 1:
        return 0.0

    bwt = 0.0
    for i in range(T - 1):
        R_T_i = phase_performances[-1][i]  # performance after all phases
        R_i_i = phase_performances[i][i]   # performance right after phase i
        bwt += R_T_i - R_i_i

    bwt /= (T - 1)
    return bwt


def compute_video_level_ap(video_scores, video_labels, num_classes):
    """
    Compute multi-class video-level AP for XD-Violence.

    Args:
        video_scores: [N, C] predicted class scores
        video_labels: [N] ground truth class indices
        num_classes: total number of classes

    Returns:
        class_aps: per-class AP values
        mean_ap: mean AP across all classes
    """
    class_aps = []
    for c in range(num_classes):
        # Binary labels for class c
        binary_labels = (video_labels == c).astype(int)
        if binary_labels.sum() == 0:
            class_aps.append(0.0)
            continue
        if binary_labels.sum() == len(binary_labels):
            class_aps.append(1.0)
            continue
        ap = compute_ap(video_scores[:, c], binary_labels)
        class_aps.append(ap)

    mean_ap = np.mean(class_aps)
    return class_aps, mean_ap


def compute_video_level_auc(video_scores, video_labels, num_classes):
    """
    Compute multi-class video-level AUC for UCF-Crime using the trapezoidal
    rule per paper Eq. 26.

    Args:
        video_scores: [N, C] predicted class scores
        video_labels: [N] ground truth class indices
        num_classes: total number of classes

    Returns:
        class_aucs: per-class AUC values
        mean_auc: mean AUC across all classes
    """
    class_aucs = []
    for c in range(num_classes):
        binary_labels = (video_labels == c).astype(int)
        if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
            class_aucs.append(0.5)
            continue
        # Eq. 26: trapezoidal-rule AUC, not sklearn's roc_auc_score
        auc = compute_auc_trapezoidal(video_scores[:, c], binary_labels)
        class_aucs.append(auc)

    mean_auc = np.mean(class_aucs)
    return class_aucs, mean_auc
