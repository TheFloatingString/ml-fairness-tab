"""Subgroup and disparity metrics, computed via fairlearn.

Kept separate from inference so the same metrics apply uniformly to TabPFN,
MLP, and XGBoost outputs regardless of which hardware/dtype config produced
them.

``summarize()`` takes a DataFrame of sensitive attributes (``gender`` /
``race`` / ``age``, whichever the dataset exposes) and, for *each* attribute,
reports the three standard group-fairness criteria as a per-group rate plus a
max-min disparity. Every disparity comes with:

  - ``diff``      -- the max-min gap of the underlying rate (already >= 0);
  - ``abs_diff``  -- ``abs(diff)``, kept explicit so downstream code is robust
                     if the gap definition ever becomes signed;
  - ``diff_se``   -- Wald (normal-approximation) standard error of the gap,
                     propagated from the two extreme groups' binomial
                     proportion SEs: ``sqrt(p_hi(1-p_hi)/n_hi + p_lo(1-p_lo)/n_lo)``,
                     where ``n`` is the group's total count for demographic
                     parity, its positive count for equal opportunity / the
                     TPR term, and its negative count for the FPR term.

  - demographic parity  -> per-group selection rate  + gap (diff/abs/se)
  - equal opportunity    -> per-group TPR              + gap (diff/abs/se)
  - equalized odds       -> per-group TPR & FPR        + gap = max(TPR gap,
                            FPR gap) (fairlearn's definition), with the
                            TPR and FPR components reported separately.

The flat top-level keys (``accuracy``, ``demographic_parity_diff``,
``equalized_odds_diff``, ``*_diff_abs``, ``*_diff_se``, ``subgroup_*``) are
kept for the primary attribute so analyze.py / the web dashboard keep
working; the full per-attribute breakdown lives under ``by_attribute``.

The Wald SE is a normal approximation; on the smallest split (German Credit,
~200 test rows) treat it as indicative rather than exact.
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score

# Primary attribute per dataset: the one the flat legacy keys point at.
_PRIMARY_ORDER = ["gender", "age", "race"]


def _grp(series) -> dict[str, float]:
    return {str(k): float(v) for k, v in series.items()}


def _prop_se(p: float, n: float) -> float:
    """Wald standard error of a binomial proportion ``p`` estimated from ``n``
    independent trials. ``nan`` when the rate is undefined or ``n <= 0``."""
    if not np.isfinite(p) or not np.isfinite(n) or n <= 0:
        return float("nan")
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def _n(n_by_group: pd.Series, g) -> float:
    v = n_by_group.get(g)
    return float(v) if v is not None and np.isfinite(v) else 0.0


def _gap(rate_by_group: pd.Series, n_by_group: pd.Series) -> dict:
    """max-min gap of a per-group rate, with a Wald SE propagated from the two
    extreme groups. Groups whose rate is ``nan`` (e.g. no positives) are
    dropped, matching fairlearn's ``MetricFrame.difference``.

    Returns ``{diff, abs_diff, diff_se, hi_group, lo_group, by_group_se}``."""
    s = rate_by_group.dropna()
    by_group_se = {
        str(g): _prop_se(float(s[g]), _n(n_by_group, g)) for g in s.index
    }
    if s.empty:
        nan = float("nan")
        return {
            "diff": nan, "abs_diff": nan, "diff_se": nan,
            "hi_group": None, "lo_group": None, "by_group_se": by_group_se,
        }
    hi, lo = s.idxmax(), s.idxmin()
    gap = float(s[hi] - s[lo])
    se = float(
        np.hypot(
            _prop_se(float(s[hi]), _n(n_by_group, hi)),
            _prop_se(float(s[lo]), _n(n_by_group, lo)),
        )
    )
    return {
        "diff": gap,
        "abs_diff": abs(gap),
        "diff_se": se,
        "hi_group": str(hi),
        "lo_group": str(lo),
        "by_group_se": by_group_se,
    }


def _summarize_one(y_true: np.ndarray, y_pred: np.ndarray, s: pd.Series) -> dict:
    # fairlearn rejects a non-string Series name.
    s = pd.Series(np.asarray(s), name=str(s.name if s.name is not None else "group"))
    yt = np.asarray(y_true)

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "count": count,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=s,
    )

    # Per-group denominators for the SEs: total for demographic parity,
    # positives for TPR / equal opportunity, negatives for FPR.
    g = pd.DataFrame({"y": yt, "s": s.to_numpy()})
    n_all = mf.by_group["count"]
    n_pos = g.groupby("s")["y"].apply(lambda v: float(np.sum(np.asarray(v) == 1)))
    n_neg = g.groupby("s")["y"].apply(lambda v: float(np.sum(np.asarray(v) == 0)))

    acc = _gap(mf.by_group["accuracy"], n_all)
    dp = _gap(mf.by_group["selection_rate"], n_all)
    eopp = _gap(mf.by_group["tpr"], n_pos)
    fpr_gap = _gap(mf.by_group["fpr"], n_neg)
    # Equalized odds = the larger of the TPR gap and the FPR gap (fairlearn's
    # definition); its SE is that of whichever component is larger.
    eodds = eopp if eopp["abs_diff"] >= fpr_gap["abs_diff"] else fpr_gap

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "subgroup_accuracy": _grp(mf.by_group["accuracy"]),
        "subgroup_accuracy_se": acc["by_group_se"],
        "subgroup_count": {k: int(v) for k, v in _grp(n_all).items()},
        "accuracy_diff": acc["diff"],
        "accuracy_diff_abs": acc["abs_diff"],
        "accuracy_diff_se": acc["diff_se"],
        "demographic_parity": {
            "by_group": _grp(mf.by_group["selection_rate"]),
            "by_group_se": dp["by_group_se"],
            "diff": dp["diff"],
            "abs_diff": dp["abs_diff"],
            "diff_se": dp["diff_se"],
        },
        "equal_opportunity": {
            "by_group": _grp(mf.by_group["tpr"]),
            "by_group_se": eopp["by_group_se"],
            "diff": eopp["diff"],
            "abs_diff": eopp["abs_diff"],
            "diff_se": eopp["diff_se"],
        },
        "equalized_odds": {
            "by_group_tpr": _grp(mf.by_group["tpr"]),
            "by_group_fpr": _grp(mf.by_group["fpr"]),
            "by_group_tpr_se": eopp["by_group_se"],
            "by_group_fpr_se": fpr_gap["by_group_se"],
            "tpr_diff": eopp["diff"],
            "tpr_diff_se": eopp["diff_se"],
            "fpr_diff": fpr_gap["diff"],
            "fpr_diff_se": fpr_gap["diff_se"],
            "diff": eodds["diff"],
            "abs_diff": eodds["abs_diff"],
            "diff_se": eodds["diff_se"],
        },
    }


def _as_frame(sensitive) -> pd.DataFrame:
    if isinstance(sensitive, pd.DataFrame):
        df = sensitive.reset_index(drop=True)
        df.columns = [str(c) for c in df.columns]
        return df
    s = pd.Series(sensitive).reset_index(drop=True)
    return s.to_frame(name=s.name or "group")


def correct_vector(y_true, y_pred) -> list[int]:
    """Per-row correctness: ``1`` where the prediction matches the label,
    ``0`` otherwise. The single definition of the "1 = correct" convention
    used by the prediction logs."""
    return (np.asarray(y_true) == np.asarray(y_pred)).astype(int).tolist()


def sensitive_labels(sensitive) -> dict[str, list[str]]:
    """Index-aligned string labels for every sensitive attribute the dataset
    exposes (``gender`` / ``race`` / ``age``), as ``{attr: [label, ...]}``."""
    df = _as_frame(sensitive)
    return {col: df[col].astype(str).tolist() for col in df.columns}


def summarize(y_true: np.ndarray, y_pred: np.ndarray, sensitive) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sens = _as_frame(sensitive)

    by_attribute = {
        col: _summarize_one(y_true, y_pred, sens[col]) for col in sens.columns
    }
    primary = next((a for a in _PRIMARY_ORDER if a in by_attribute), sens.columns[0])
    p = by_attribute[primary]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "primary_attribute": primary,
        "by_attribute": by_attribute,
        # ---- flat legacy keys (primary attribute) ----
        "subgroup_accuracy": p["subgroup_accuracy"],
        "subgroup_selection_rate": p["demographic_parity"]["by_group"],
        "subgroup_tpr": p["equalized_odds"]["by_group_tpr"],
        "subgroup_fpr": p["equalized_odds"]["by_group_fpr"],
        "subgroup_count": p["subgroup_count"],
        "accuracy_diff": p["accuracy_diff"],
        "accuracy_diff_abs": p["accuracy_diff_abs"],
        "accuracy_diff_se": p["accuracy_diff_se"],
        "demographic_parity_diff": p["demographic_parity"]["diff"],
        "demographic_parity_diff_abs": p["demographic_parity"]["abs_diff"],
        "demographic_parity_diff_se": p["demographic_parity"]["diff_se"],
        "equal_opportunity_diff": p["equal_opportunity"]["diff"],
        "equal_opportunity_diff_abs": p["equal_opportunity"]["abs_diff"],
        "equal_opportunity_diff_se": p["equal_opportunity"]["diff_se"],
        "equalized_odds_diff": p["equalized_odds"]["diff"],
        "equalized_odds_diff_abs": p["equalized_odds"]["abs_diff"],
        "equalized_odds_diff_se": p["equalized_odds"]["diff_se"],
    }
