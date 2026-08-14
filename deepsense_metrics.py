"""
Evaluation metrics from the DeepSense 2023 V2V Beam Prediction Challenge baseline.

Reference: https://github.com/DeepSense6G/Multi-Modal-V2V-Beam-Prediction-Challenge-2023-Baseline
"""

from __future__ import annotations

import numpy as np


N_BEAMS_TOTAL = 256


def circular_distance(a: int, b: int, n: int = N_BEAMS_TOTAL) -> float:
    """Shortest distance between beam indices on a circular codebook."""
    a = int(a) % n
    b = int(b) % n
    dist = abs(a - b)
    return min(dist, n - dist)


def compute_topk_accuracy(
    ranked_beams: np.ndarray,
    true_beam: np.ndarray,
    top_k: list[int] | None = None,
) -> dict[int, float]:
    """
    Top-k hit rate: true beam appears in the first k ranked predictions.

    ranked_beams: (N, K) beam indices sorted by model confidence (best first)
    true_beam: (N,) ground-truth best beam index
    """
    if top_k is None:
        top_k = [1, 3, 5, 13]
    n = len(true_beam)
    hits = {}
    for k in top_k:
        hit = np.any(ranked_beams[:, :k] == true_beam.reshape(-1, 1), axis=1)
        hits[k] = float(np.mean(hit))
    return hits


def average_power_loss(
    true_best_pwr: np.ndarray,
    est_best_pwr: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    APL (Average Power Loss) in dB — official competition metric.

    Measures power wasted by using the predicted beam instead of the optimum.
    Lower is better; 0 dB means perfect beam selection.
    """
    ratio = (est_best_pwr + eps) / (true_best_pwr + eps)
    return float(np.mean(10.0 * np.log10(ratio)))


def power_ratio(true_best_pwr: np.ndarray, est_best_pwr: np.ndarray, eps: float = 1e-12) -> float:
    """Average received-power ratio (linear scale). Used in Mollah et al. papers."""
    return float(np.mean((est_best_pwr + eps) / (true_best_pwr + eps)))


def summarize_predictions(
    pred_beam: np.ndarray,
    true_beam: np.ndarray,
    power_profile: np.ndarray,
    pred_logits: np.ndarray | None = None,
    top_k: list[int] | None = None,
) -> dict:
    """
    Full metric summary for a batch of predictions.

    pred_beam: (N,) top-1 predicted beam indices
    true_beam: (N,) ground-truth best beam indices
    power_profile: (N, 256) linear received power per beam
    pred_logits: (N, 256) model output logits or ranking scores
    """
    if top_k is None:
        top_k = [1, 3, 5, 13]

    pred_beam = pred_beam.astype(int)
    true_beam = true_beam.astype(int)
    n = len(true_beam)

    true_best_pwr = power_profile[np.arange(n), true_beam]
    est_best_pwr = power_profile[np.arange(n), pred_beam]

    beam_dist = np.array([circular_distance(p, t) for p, t in zip(pred_beam, true_beam)])

    if pred_logits is not None:
        ranked = np.argsort(pred_logits, axis=1)[:, ::-1]
    else:
        ranked = np.argsort(power_profile, axis=1)[:, ::-1]
    topk_acc = compute_topk_accuracy(ranked, true_beam, top_k)

    return {
        "top1_acc": float(np.mean(pred_beam == true_beam)),
        "topk_accuracy": topk_acc,
        "avg_beam_index_distance": float(np.mean(beam_dist)),
        "apl_db": average_power_loss(true_best_pwr, est_best_pwr),
        "power_ratio": power_ratio(true_best_pwr, est_best_pwr),
    }
