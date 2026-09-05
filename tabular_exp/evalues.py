"""E-values for cross-hardware drift testing.

An **e-value** ``E`` for a null ``H0`` is a non-negative statistic with
``E_{H0}[E] <= 1``. Then ``1/E`` is a conservative p-value, and -- the reason
we use them here -- e-values for independent experiments **multiply** (the
product is still an e-value for the intersection null) and the underlying
capital processes are non-negative supermartingales, so they stay valid under
optional stopping / continuation (accruing more rows, GPUs, or datasets).

Two nulls are tested, each for a GPU ``g`` against a reference GPU (T4):

* **metric level** -- ``H0``: ``g`` and the reference induce the same fairness
  metric on the test distribution. Paired per-row test: same rows, same frozen
  weights, only the hardware differs, so the per-row prediction difference
  ``d_i = hat y_i^g - hat y_i^ref`` has mean 0 under ``H0``.

* **logit level** -- ``H0``: ``g`` and the reference produce the same
  softmax-input (row-centered log-proba) per row, up to the model's own
  within-GPU nondeterminism, which is estimated from ``repeats`` independent
  forward passes.

Constructions
-------------
``betting_evalue`` -- grid-mixture betting e-value (Waudby-Smith & Ramdas,
"Estimating means of bounded random variables by betting", 2023) for the mean
of a ``[0, 1]`` variable; ``alpha``-free, two-sided.

``gaussian_mixture_evalue`` -- mixture e-value for ``z_i ~iid N(0, 1)`` against
a ``N(theta, 1)``, ``theta ~ N(0, tau^2)`` shift.

``ebh`` -- e-Benjamini-Hochberg (Wang & Ramdas, 2022): FDR control over a grid
of e-values under arbitrary dependence.
"""

import numpy as np


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(a - m))))


def _mix_exp(logs) -> float:
    """Mean of ``exp(logs)`` -- an average of log-scale e-values -- returning
    ``inf`` instead of overflowing when the evidence is overwhelming."""
    logs = np.asarray(logs, float)
    m = float(np.max(logs))
    if m > 700.0:
        return float("inf")
    return float(np.mean(np.exp(logs)))


def betting_evalue(y, mu0: float, n_lambda: int = 24) -> float:
    """Grid-mixture betting e-value for ``H0: E[Y] = mu0``, ``Y in [0, 1]``
    (two-sided).

    Capital process for a fixed bet ``lam`` is
    ``K_n(lam) = prod_i (1 + lam (Y_i - mu0))`` -- a non-negative martingale
    with ``E_{H0}[K_n] = 1``. We average ``K_n(lam)`` over a fixed grid of
    ``lam`` spanning the widest range that keeps every factor positive
    (``-1/(1-mu0) < lam < 1/mu0``); a convex mixture of e-values is an
    e-value, and the grid needs no ``alpha``.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 1.0
    mu0 = float(np.clip(mu0, 1e-6, 1.0 - 1e-6))
    lo, hi = -0.9 / (1.0 - mu0), 0.9 / mu0
    half = max(2, n_lambda // 2)
    lam = np.concatenate(
        [np.linspace(lo, -1e-3, half), np.linspace(1e-3, hi, half)]
    )
    log_cap = np.log1p(np.outer(lam, y - mu0)).sum(axis=1)  # (J,)
    return float(np.exp(_logsumexp(log_cap) - np.log(lam.size)))


def paired_diff_evalue(a, b, mask=None) -> float:
    """E-value for ``H0: E[a_i - b_i] = 0`` over rows where ``mask`` is True,
    with ``a_i, b_i in {0, 1}`` (hard predictions of two GPUs). ``d = a - b``
    in ``{-1, 0, 1}`` is mapped to ``u = (d + 1) / 2`` and ``E[u] = 1/2`` is
    tested. Masking to a subgroup tests that subgroup's selection-rate shift."""
    a = np.asarray(a)
    b = np.asarray(b)
    if mask is not None:
        m = np.asarray(mask, bool)
        a, b = a[m], b[m]
    if a.size == 0:
        return 1.0
    u = (a.astype(float) - b.astype(float) + 1.0) / 2.0
    return betting_evalue(u, 0.5)


def mcnemar_evalue(a, b, mask=None) -> float:
    """E-value for ``H0``: among rows where ``a != b``,
    ``P(a=1, b=0) = P(a=0, b=1) = 1/2`` (prediction flips are symmetric). An
    asymmetry within a subgroup means GPU ``a`` systematically moves that
    subgroup's selection rate relative to ``b``."""
    a = np.asarray(a)
    b = np.asarray(b)
    if mask is not None:
        m = np.asarray(mask, bool)
        a, b = a[m], b[m]
    disc = a != b
    if not disc.any():
        return 1.0
    up = (a[disc] > b[disc]).astype(float)  # 1 for 0->1 flips, 0 for 1->0
    return betting_evalue(up, 0.5)


def gap_drift_evalue(a, b, group, hi, lo, mask=None) -> float:
    """E-value for ``H0``: the *signed* rate gap ``rate(hi) - rate(lo)`` is
    unchanged between predictions ``a`` and ``b`` (``rate`` = selection rate,
    or TPR/FPR when ``mask`` restricts to ``y=1`` / ``y=0``).

    Per-row influence of the between-GPU gap difference,
    ``psi_i = 1[g_i=hi]/pi_hi (a_i - b_i) - 1[g_i=lo]/pi_lo (a_i - b_i)``,
    is an unbiased bounded estimator of the gap change, so ``H0 <=> E[psi]=0``.
    ``hi`` / ``lo`` must come from the *reference* GPU (not ``a``) to avoid
    selecting the groups on the very quantity under test."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    group = np.asarray(group)
    if mask is not None:
        m = np.asarray(mask, bool)
        a, b, group = a[m], b[m], group[m]
    if a.size == 0:
        return 1.0
    d = a - b
    in_hi, in_lo = group == hi, group == lo
    pi_hi, pi_lo = float(in_hi.mean()), float(in_lo.mean())
    if pi_hi == 0.0 or pi_lo == 0.0:
        return 1.0
    psi = np.where(in_hi, d / pi_hi, 0.0) - np.where(in_lo, d / pi_lo, 0.0)
    bound = 1.0 / min(pi_hi, pi_lo)  # |psi_i| <= bound
    u = np.clip((psi + bound) / (2.0 * bound), 0.0, 1.0)
    return betting_evalue(u, 0.5)


def gaussian_scale_evalue(z, tau2_grid=(1.0, 4.0, 16.0)) -> float:
    """E-value for ``H0: z_i ~iid N(0, 1)`` against a variance-inflation
    alternative ``z_i ~ N(0, 1 + tau^2)`` -- picks up *any* structured per-row
    shift (its magnitude, regardless of sign). Per row and ``tau^2``,
    ``e_i = (1+tau^2)^{-1/2} exp(z_i^2 tau^2 / (2(1+tau^2)))``,
    ``E_{H0}[e_i] = 1``; row product, then mean over the grid (convex mixture).
    Powered when typical ``|z_i| >~ 1`` (drift comparable to the noise SD)."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 1.0
    ss = float(np.sum(z ** 2))
    return _mix_exp(
        [z.size * (-0.5 * np.log1p(t2)) + 0.5 * (t2 / (1.0 + t2)) * ss for t2 in tau2_grid]
    )


def gaussian_bias_evalue(z, tau2_grid=(0.25, 1.0, 4.0)) -> float:
    """E-value for ``H0: z_i ~iid N(0, 1)`` against a *net directional* shift
    ``E[z_i] = theta``, ``theta ~ N(0, tau^2)``. Mixing over the whole
    sequence gives
    ``E = (n tau^2 + 1)^{-1/2} exp( S_n^2 / (2(n + 1/tau^2)) )``, ``S_n =
    sum z_i``, with ``E_{H0}[E] = 1``; mean over the grid. Detects an
    arbitrarily small consistent bias given enough rows (unlike the scale
    test)."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    n = z.size
    if n == 0:
        return 1.0
    sn2 = float(np.sum(z)) ** 2
    return _mix_exp(
        [-0.5 * np.log(n * t2 + 1.0) + sn2 / (2.0 * (n + 1.0 / t2)) for t2 in tau2_grid]
    )


def logit_drift_evalue(z) -> float:
    """Headline logit-level e-value: mean of the scale and bias e-values (a
    convex mixture, still an e-value for ``H0: no drift``). ``z`` is the
    per-row logit shift divided by its within-GPU noise SD."""
    return 0.5 * (gaussian_scale_evalue(z) + gaussian_bias_evalue(z))


def combine_evalues(evalues) -> float:
    """Product of e-values for independent experiments -- still an e-value for
    the null that *none* of them drift. ``None`` / ``NaN`` entries are
    dropped (treated as an uninformative ``E = 1``)."""
    vals = [
        float(v)
        for v in evalues
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    return float(np.prod(vals)) if vals else 1.0


def ebh(evalues, alpha: float) -> np.ndarray:
    """e-Benjamini-Hochberg (Wang & Ramdas, 2022). Given e-values for ``K``
    hypotheses, return a boolean rejection mask controlling FDR at ``alpha``
    under arbitrary dependence: reject the ``k*`` largest e-values where
    ``k* = max{k : E_(k) >= K / (alpha k)}``."""
    e = np.asarray(evalues, float)
    e = np.where(np.isnan(e), 0.0, e)
    K = e.size
    if K == 0:
        return np.zeros(0, bool)
    order = np.argsort(-e)
    sorted_e = e[order]
    thresh = K / (alpha * np.arange(1, K + 1))
    ok = np.where(sorted_e >= thresh)[0]
    kstar = int(ok.max()) + 1 if ok.size else 0
    rej = np.zeros(K, bool)
    rej[order[:kstar]] = True
    return rej
