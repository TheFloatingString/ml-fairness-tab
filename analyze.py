"""Cross-hardware fairness-variance analysis over modal_app.py's results/results.json.

Core question: for each model, how much does per-subgroup accuracy and
disparity (demographic parity / equal opportunity / equalized odds, plus
per-group selection rate / TPR / FPR from fairlearn's MetricFrame) move just
from changing GPU type or compute dtype, with everything else (weights, data,
seed) held fixed? XGBoost's spread is the near-zero control baseline.

Every disparity is carried as the max-min gap (``*_diff``), its absolute
value (``*_absdiff`` / ``*_diff_abs``), and a Wald standard error
(``*_se`` / ``*_diff_se``) -- see tabular_exp/fairness.py.

T4 is the reference GPU: `t4_baseline_report` shows every other GPU's metric
as a delta from the T4 run. Disparities are reported per sensitive attribute
(gender / race / age) via the `by_attribute` block.

`drift_report` summarises results/drift_correlation-<model>.json: the Pearson
R between per-(GPU, dataset) logit drift from T4 and the demographic-parity
shift from T4.
"""

import glob
import json
import os

import pandas as pd

_ID_COLS = ("model", "gpu", "dtype", "device_name")
# The three fairness criteria, as they appear under by_attribute[attr].
_CRITERIA = {
    "dp": "demographic_parity",
    "eo": "equal_opportunity",
    "eqodds": "equalized_odds",
}


def load_results(path: str = "results/results.json") -> pd.DataFrame:
    with open(path) as f:
        rows = json.load(f)

    records = []
    for row in rows:
        rec = {
            "model": row["model"],
            "gpu": row["gpu"],
            "dtype": row["dtype"],
            "device_name": row["device_name"],
            "accuracy": row["accuracy"],
            "accuracy_diff": row.get("accuracy_diff"),
            "accuracy_diff_se": row.get("accuracy_diff_se"),
            "demographic_parity_diff": row.get("demographic_parity_diff"),
            "demographic_parity_diff_se": row.get("demographic_parity_diff_se"),
            "equal_opportunity_diff": row.get("equal_opportunity_diff"),
            "equal_opportunity_diff_se": row.get("equal_opportunity_diff_se"),
            "equalized_odds_diff": row.get("equalized_odds_diff"),
            "equalized_odds_diff_se": row.get("equalized_odds_diff_se"),
        }
        # fairlearn MetricFrame subgroup breakdowns for the primary attribute.
        for key, prefix in [
            ("subgroup_accuracy", "acc"),
            ("subgroup_selection_rate", "sel"),
            ("subgroup_tpr", "tpr"),
            ("subgroup_fpr", "fpr"),
        ]:
            for group, val in row.get(key, {}).items():
                rec[f"{prefix}_{group}"] = val
        # Per-attribute disparities: gap, |gap|, and Wald SE for each of the
        # 3 criteria.
        for attr, blk in row.get("by_attribute", {}).items():
            for short, crit in _CRITERIA.items():
                c = blk.get(crit, {})
                rec[f"{short}_diff__{attr}"] = c.get("diff")
                rec[f"{short}_absdiff__{attr}"] = c.get("abs_diff")
                rec[f"{short}_se__{attr}"] = c.get("diff_se")
        records.append(rec)
    return pd.DataFrame(records)


def _metric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _ID_COLS]


def variance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Range (max-min) of each metric across all (gpu, dtype) configs, per model."""
    cols = _metric_cols(df)
    return df.groupby("model")[cols].agg(lambda s: s.max() - s.min())


def t4_baseline_report(df: pd.DataFrame) -> pd.DataFrame:
    """Each (model, dtype, gpu) row's metrics as a delta from that model's
    T4 run at the same dtype. T4 rows are all zero by construction."""
    cols = _metric_cols(df)
    out = []
    for (model, dtype), grp in df.groupby(["model", "dtype"]):
        ref = grp[grp["gpu"] == "T4"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for _, r in grp.iterrows():
            rec = {"model": model, "dtype": dtype, "gpu": r["gpu"]}
            for c in cols:
                if pd.notna(r[c]) and pd.notna(ref[c]):
                    rec[f"d_{c}"] = r[c] - ref[c]
            out.append(rec)
    return pd.DataFrame(out)


def drift_report(results_dir: str = "results") -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "drift_correlation-*.json"))):
        with open(path) as f:
            d = json.load(f)
        rows.append(
            {
                "model": d["model"],
                "n_points": d["n_points"],
                "pearson_r": d["pearson_r"],
                "p_value": d["p_value"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
    df = load_results(path)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    print("\nCross-hardware/precision range per model (max - min):")
    print(variance_report(df).to_string())

    print("\nDelta from the T4 run (per model x dtype):")
    print(t4_baseline_report(df).to_string(index=False))

    drift = drift_report(os.path.dirname(path) or ".")
    if not drift.empty:
        print("\nLogit-drift vs demographic-parity-shift correlation (T4 reference):")
        print(drift.to_string(index=False))
