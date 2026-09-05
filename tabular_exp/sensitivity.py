"""Decision-margin and input-noise-sensitivity analysis for TabPFN.

TabPFN has no training-time gradient step, so the original paper's
gradient-flow/sharp-minima story doesn't port directly. This module tests
the natural analog: is a subgroup's prediction closer to the decision
boundary (small margin), and does that make it more likely to flip under
tiny input perturbations of roughly float32-precision scale? Both are
single-hardware, single-run computations — no multi-GPU sweep needed,
so this isolates the *mechanism* from the hardware confound.
"""

import numpy as np
import pandas as pd


def decision_margins(proba: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 predicted class probability, per sample."""
    sorted_proba = np.sort(proba, axis=1)
    return sorted_proba[:, -1] - sorted_proba[:, -2]


def baseline_flip_rate(
    X_test: np.ndarray,
    baseline_preds: np.ndarray,
    predict_fn,
    n_repeats: int,
) -> np.ndarray:
    """Repeats predict_fn on the *unperturbed* X_test and measures how often
    it disagrees with itself. Isolates flips caused by the model's own
    internal stochasticity (e.g. TabPFN's permutation ensembling) from
    flips caused by injected input noise — without this control, a flat
    flip-rate-vs-noise-scale curve is ambiguous between the two causes."""
    flips = np.zeros(len(X_test))
    for _ in range(n_repeats):
        repeat_preds = predict_fn(X_test)
        flips += repeat_preds != baseline_preds
    return flips / n_repeats


def noise_flip_rates(
    X_test: np.ndarray,
    baseline_preds: np.ndarray,
    predict_fn,
    X_train_for_scale: np.ndarray,
    sigma_fracs: list[float],
    n_repeats: int,
    seed: int,
) -> dict[float, np.ndarray]:
    """For each relative noise level (as a fraction of each feature's
    training-set std), repeatedly perturb X_test and measure per-sample
    prediction flip rate against the clean baseline."""
    feature_std = X_train_for_scale.std(axis=0, keepdims=True)
    feature_std[feature_std == 0] = 1.0

    rng = np.random.default_rng(seed)
    out = {}
    for sigma_frac in sigma_fracs:
        flips = np.zeros(len(X_test))
        for _ in range(n_repeats):
            noise = rng.normal(0, 1, size=X_test.shape).astype(np.float32) * feature_std * sigma_frac
            noisy_preds = predict_fn(X_test + noise)
            flips += noisy_preds != baseline_preds
        out[sigma_frac] = flips / n_repeats
    return out


def subgroup_means(values: np.ndarray, sensitive: pd.Series) -> dict[str, float]:
    out = {}
    for group in sensitive.unique():
        mask = (sensitive == group).to_numpy()
        out[str(group)] = float(np.asarray(values)[mask].mean())
    return out
