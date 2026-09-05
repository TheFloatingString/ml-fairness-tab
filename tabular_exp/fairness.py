"""Subgroup and disparity metrics, computed via fairlearn.

Kept separate from inference so the same metrics apply uniformly to TabPFN,
MLP, and XGBoost outputs regardless of which hardware/dtype config produced
them.

``summarize()`` takes a DataFrame of sensitive attributes (``gender`` /
``race`` / ``age``, whichever the dataset exposes) and, for *each* attribute,
reports the three standard fairness criteria as both an absolute per-group
value and a disparity (difference):

  - demographic parity   -> per-group selection rate      + max-min diff
  - equal opportunity     -> per-group TPR                  + max-min diff
  - equalized odds        -> per-group TPR & FPR            + fairlearn diff

The flat top-level keys (``accuracy``, ``demographic_parity_diff``,
``equalized_odds_diff``, ``subgroup_*``) are kept for the primary attribute
so analyze.py / the web dashboard keep working; the full breakdown lives
under ``by_attribute``.
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score

# Primary attribute per dataset: the one the flat legacy keys point at.
_PRIMARY_ORDER = ["gender", "age", "race"]


def _grp(series) -> dict[str, float]:
    return {str(k): float(v) for k, v in series.items()}


def _summarize_one(y_true: np.ndarray, y_pred: np.ndarray, s: pd.Series) -> dict:
    # fairlearn rejects a non-string Series name.
    s = pd.Series(np.asarray(s), name=str(s.name if s.name is not None else "group"))
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
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "subgroup_accuracy": _grp(mf.by_group["accuracy"]),
        "subgroup_count": {k: int(v) for k, v in _grp(mf.by_group["count"]).items()},
        "accuracy_diff": float(mf.difference()["accuracy"]),
        "demographic_parity": {
            "by_group": _grp(mf.by_group["selection_rate"]),
            "diff": float(
                demographic_parity_difference(y_true, y_pred, sensitive_features=s)
            ),
        },
        "equal_opportunity": {
            "by_group": _grp(mf.by_group["tpr"]),
            "diff": float(mf.difference()["tpr"]),
        },
        "equalized_odds": {
            "by_group_tpr": _grp(mf.by_group["tpr"]),
            "by_group_fpr": _grp(mf.by_group["fpr"]),
            "diff": float(
                equalized_odds_difference(y_true, y_pred, sensitive_features=s)
            ),
        },
    }


def _as_frame(sensitive) -> pd.DataFrame:
    if isinstance(sensitive, pd.DataFrame):
        df = sensitive.reset_index(drop=True)
        df.columns = [str(c) for c in df.columns]
        return df
    s = pd.Series(sensitive).reset_index(drop=True)
    return s.to_frame(name=s.name or "group")


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
        "demographic_parity_diff": p["demographic_parity"]["diff"],
        "equal_opportunity_diff": p["equal_opportunity"]["diff"],
        "equalized_odds_diff": p["equalized_odds"]["diff"],
    }
