"""
Statistical metrics for SleepFM Interpretability Project.

Provides C-Index, AUROC, and bootstrap confidence interval calculations.

Design decision (Karar 4): Manual bootstrap implementation using numpy.
No scipy.stats.bootstrap — full transparency over every resampling step.
BCa (bias-corrected accelerated) is deferred; start with percentile method.
"""

import warnings

import numpy as np
from lifelines.utils import concordance_index as _lifelines_ci
from typing import Tuple, Callable, Optional

from config import (
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    PERMUTATION_N,
)


def concordance_index(
    event_times: np.ndarray,
    predicted_scores: np.ndarray,
    event_indicators: np.ndarray,
) -> float:
    """Compute concordance index (C-Index) for survival predictions.

    C-Index measures the model's ability to correctly rank patients
    by risk. 0.5 = random, 1.0 = perfect discrimination.

    Args:
        event_times: Time to event or censoring, shape (n_samples,).
        predicted_scores: Model's risk predictions, shape (n_samples,).
            Higher score = higher predicted risk.
        event_indicators: 1 = event occurred, 0 = censored, shape (n_samples,).

    Returns:
        C-Index value between 0 and 1.

    Notes:
        - Uses lifelines.utils.concordance_index internally
        - Wraps the lifelines call so the rest of the project
          doesn't need to import lifelines directly
    """
    return float(_lifelines_ci(event_times, predicted_scores, event_indicators))


def auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute Area Under ROC Curve.

    Args:
        y_true: Binary labels (0 or 1), shape (n_samples,).
        y_score: Predicted probabilities or scores, shape (n_samples,).

    Returns:
        AUROC value between 0 and 1. 0.5 = random, 1.0 = perfect.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError(
            f"y_true must contain only 0 and 1, got unique values: {np.unique(y_true)}"
        )
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort samples by descending predicted score
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # Cumulative TP and FP counts at each position
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    # Tie handling: keep only the last point of each group of identical scores.
    # np.diff on a descending array is <= 0; != 0 marks a score change.
    mask = np.concatenate([np.diff(y_score_sorted) != 0, [True]])
    tps = np.concatenate([[0], tps[mask]])
    fps = np.concatenate([[0], fps[mask]])

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Trapezoidal integration of ROC curve
    return float(np.trapz(tpr, fpr))


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[..., float],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    metric_args: Tuple[np.ndarray, ...] = (),
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Manual implementation using numpy (Karar 4 — Secenek A).
    Uses percentile method. BCa deferred to later if needed.

    Algorithm:
    1. Compute point estimate on full data
    2. For each resample iteration:
       a. Sample indices with replacement
       b. Compute metric on resampled data
    3. Take percentiles of bootstrap distribution for CI bounds

    Args:
        y_true: Ground truth values, shape (n_samples,) or (n_samples, n_features).
        y_pred: Predicted values, same shape as y_true.
        metric_fn: Callable with signature metric_fn(y_true, y_pred, *metric_args) -> float.
            For 2-arg metrics (e.g. auroc): use as-is.
            For 3-arg metrics (e.g. concordance_index): pass extra arrays via metric_args.
        metric_args: Extra array arguments forwarded to metric_fn after y_true and y_pred.
            Each element must be shape (n_samples, ...) — resampled with the same indices.
            Example: bootstrap_ci(times, scores, concordance_index,
                                   metric_args=(event_indicators,))
        n_resamples: Number of bootstrap iterations.
            Defaults to BOOTSTRAP_N_RESAMPLES from config.
        confidence_level: CI level (e.g. 0.95 for 95% CI).
            Defaults to BOOTSTRAP_CONFIDENCE_LEVEL from config.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
        - point_estimate: metric on full (non-resampled) data
        - ci_lower: lower bound of confidence interval
        - ci_upper: upper bound of confidence interval

    Notes:
        - You can inspect the full bootstrap distribution by modifying
          this function to also return the array of bootstrap values
        - NaN handling: if a resample produces NaN, it is excluded
          from the distribution. If >10% of resamples are NaN, a warning
          is raised. If ALL resamples are NaN, returns
          (point_estimate, nan, nan) with a warning.
        - Small sample awareness: if n_samples < 30, CI may be unreliable.
          This is intentionally NOT hidden — you should see it.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)

    if n_samples < 30:
        warnings.warn(
            f"n_samples={n_samples} < 30. Bootstrap CI may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Step 1: Point estimate on full data
    point_estimate = metric_fn(y_true, y_pred, *metric_args)

    # Step 2: Resample with replacement and collect bootstrap distribution
    rng = np.random.default_rng(seed)
    bootstrap_values: list = []
    nan_count = 0

    for _ in range(n_resamples):
        indices = rng.integers(0, n_samples, size=n_samples)
        val = metric_fn(y_true[indices], y_pred[indices], *(a[indices] for a in metric_args))
        if np.isnan(val):
            nan_count += 1
        else:
            bootstrap_values.append(val)

    if nan_count > 0.1 * n_resamples:
        warnings.warn(
            f"{nan_count}/{n_resamples} bootstrap resamples produced NaN "
            f"({100 * nan_count / n_resamples:.1f}% > 10%). CI may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Step 3: Percentile CI bounds
    if len(bootstrap_values) == 0:
        warnings.warn(
            "All bootstrap resamples produced NaN. Cannot compute CI.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(point_estimate), float("nan"), float("nan")
    boot_arr = np.array(bootstrap_values)
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_arr, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_arr, 100 * (1.0 - alpha / 2)))

    return float(point_estimate), ci_lower, ci_upper


def permutation_test(
    scores_baseline: np.ndarray,
    scores_ablated: np.ndarray,
    n_permutations: int = PERMUTATION_N,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Two-sample permutation test for ablation significance.

    Tests H0: ablating a modality has no effect on prediction scores.

    Algorithm:
    1. Compute observed difference: mean(baseline) - mean(ablated)
    2. For each permutation:
       a. Pool both arrays, shuffle, split back into two groups
       b. Compute difference of the shuffled groups
    3. p-value = fraction of permuted differences >= observed difference

    Args:
        scores_baseline: Metric values from baseline model, shape (n_samples,).
        scores_ablated: Metric values from ablated model, shape (n_samples,).
        n_permutations: Number of random permutations.
            Defaults to PERMUTATION_N from config.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (observed_difference, p_value).
        - observed_difference: mean(baseline) - mean(ablated)
        - p_value: two-sided p-value — proportion of permutations where
          |perm_diff| >= |observed_diff|. p_value < 0.05 = statistically
          significant effect in either direction.

    Notes:
        - This is a non-parametric test — no distribution assumptions
        - p_value is two-sided: counts permutations where |perm_diff| >= |observed_diff|.
          Use observed_difference to interpret direction.
        - observed_difference > 0 → baseline outperforms ablated.
          observed_difference < 0 → ablation unexpectedly improved scores.
    """
    # Step 1: Observed difference
    observed_diff = float(np.mean(scores_baseline) - np.mean(scores_ablated))

    # Step 2: Permutation distribution
    pooled = np.concatenate([scores_baseline, scores_ablated])
    n_baseline = len(scores_baseline)
    rng = np.random.default_rng(seed)
    count = 0

    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        perm_diff = np.mean(shuffled[:n_baseline]) - np.mean(shuffled[n_baseline:])
        # Two-sided: count permutations at least as extreme in either direction
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return observed_diff, float(p_value)
