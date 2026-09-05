"""E-value tests for cross-hardware drift, over the artifacts `run-quick`
writes. Post-runtime only -- no Modal, needs just numpy + pandas.

Two families, each testing GPU ``g`` against the reference GPU (T4):

  metric level  (results/predictions/predictions-*.json)
      H0: g and T4 induce the same fairness metric on the test distribution.
      Paired per-row tests -- same rows, same frozen weights, hardware only.
        * gap-drift e-value per criterion (demographic parity / equal
          opportunity / equalized odds): does the signed subgroup gap move?
        * McNemar e-value per subgroup: are g-vs-T4 prediction flips within
          the subgroup symmetric?

  logit level   (results/logits/logits-*.json)
      H0: g and T4 produce the same softmax input per row, up to noise.
      Gaussian-mixture e-value (scale + net-bias) on the per-row log-odds
      shift, standardized by the noise scale: the within-GPU SD from the
      `repeats` forward passes when they vary, else -- they are typically
      bit-identical for these models at fp32 -- an arithmetic-precision floor
      (the closest GPU's drift, min 1e-4). `argmax_flips_vs_ref` and
      `deterministic` are reported alongside.

E-values across datasets are multiplied (still an e-value for "no drift
anywhere"); e-BH controls FDR across the hypothesis grid.

Usage:
    uv run python evalue_analysis.py [--logit-only | --metric-only]
                                     [--alpha 0.05] [--ref T4] [--results-dir results]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from tabular_exp.evalues import (
    combine_evalues,
    ebh,
    gap_drift_evalue,
    gaussian_bias_evalue,
    gaussian_scale_evalue,
    logit_drift_evalue,
    mcnemar_evalue,
)

_CRITERIA = ("demographic_parity", "equal_opportunity", "equalized_odds")
_DET_TOL = 1e-9


# --------------------------------------------------------------------------- #
# metric level
# --------------------------------------------------------------------------- #
def _hard_preds(entry: dict, y_true: np.ndarray) -> np.ndarray:
    """Per-row predictions for one config: prefer the logged ``y_pred``, else
    reconstruct from ``correct`` (binary: hat y = y if correct else 1 - y)."""
    if entry.get("y_pred") is not None:
        return np.asarray(entry["y_pred"], dtype=int)
    correct = np.asarray(entry["correct"], dtype=int)
    return np.where(correct == 1, y_true, 1 - y_true)


def _group_rate(pred: np.ndarray, groups: np.ndarray, mask: np.ndarray | None):
    """{group: selection rate} for the reference predictions, over ``mask``."""
    idx = np.ones(len(pred), bool) if mask is None else np.asarray(mask, bool)
    out = {}
    for g in np.unique(groups):
        sel = idx & (groups == g)
        out[g] = float(pred[sel].mean()) if sel.any() else np.nan
    return out


def _hi_lo(rate: dict):
    items = [(g, r) for g, r in rate.items() if np.isfinite(r)]
    if len(items) < 2:
        return None, None
    hi = max(items, key=lambda kv: kv[1])[0]
    lo = min(items, key=lambda kv: kv[1])[0]
    return hi, lo


def _criterion_evalue(crit, a, b, groups, y_true):
    """gap-drift e-value for one criterion; equalized odds = mean of the TPR
    and FPR gap-drift e-values (a convex mixture, still an e-value)."""
    if crit == "demographic_parity":
        masks = [None]
    elif crit == "equal_opportunity":
        masks = [y_true == 1]
    else:  # equalized_odds -> TPR gap and FPR gap
        masks = [y_true == 1, y_true == 0]
    evs = []
    for m in masks:
        hi, lo = _hi_lo(_group_rate(b, groups, m))
        evs.append(1.0 if hi is None else gap_drift_evalue(a, b, groups, hi, lo, mask=m))
    return float(np.mean(evs))


def metric_level(results_dir: str, ref: str, alpha: float) -> dict:
    files = sorted(glob.glob(os.path.join(results_dir, "predictions", "predictions-*.json")))
    rows = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        y_true = np.asarray(d["y_true"], dtype=int)
        sens = {a: np.asarray(v) for a, v in d["sensitive"].items()}
        preds = pd.DataFrame(d["predictions"])
        for (model, dtype), grp in preds.groupby(["model", "dtype"]):
            refrow = grp[grp["gpu"] == ref]
            if refrow.empty:
                continue
            b = _hard_preds(refrow.iloc[0].to_dict(), y_true)
            for _, r in grp.iterrows():
                if r["gpu"] == ref:
                    continue
                a = _hard_preds(r.to_dict(), y_true)
                for attr, groups in sens.items():
                    for crit in _CRITERIA:
                        rows.append(
                            dict(
                                dataset=d["dataset"], model=model, dtype=dtype, gpu=r["gpu"],
                                attribute=attr, test=crit,
                                evalue=_criterion_evalue(crit, a, b, groups, y_true),
                            )
                        )
                    for g in np.unique(groups):
                        rows.append(
                            dict(
                                dataset=d["dataset"], model=model, dtype=dtype, gpu=r["gpu"],
                                attribute=attr, test=f"mcnemar[{g}]",
                                evalue=mcnemar_evalue(a, b, groups == g),
                            )
                        )
    return _aggregate(pd.DataFrame(rows), alpha, key=["model", "dtype", "gpu", "attribute", "test"])


# --------------------------------------------------------------------------- #
# logit level
# --------------------------------------------------------------------------- #
# fp32 log-odds arithmetic-noise scale: when the repeated passes are
# bit-identical (no stochastic noise) the drift e-value is taken relative to
# this floor instead. Chosen ~10x above the observed same-kernel-path
# residual (~1e-5 mean |d log-odds|) and far below a structural kernel split.
_ABS_FLOOR = 1e-4


def logit_level(results_dir: str, ref: str, alpha: float) -> dict:
    files = sorted(glob.glob(os.path.join(results_dir, "logits", "logits-*.json")))
    rows = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        if d.get("n_classes") != 2:
            rows.append(dict(dataset=d["dataset"], model=d["model"], gpu="-", test="logit_shift",
                             evalue=np.nan, note=f"n_classes={d.get('n_classes')} (only binary handled)"))
            continue
        R = int(d["repeats"])
        refg = d.get("reference_gpu", ref)
        lp = {g: np.asarray(v, dtype=np.float64) for g, v in d["logprob"].items()}  # (R, N, 2)
        odds = {g: v[:, :, 1] - v[:, :, 0] for g, v in lp.items()}                  # (R, N)
        rbar = {g: v.mean(axis=0) for g, v in odds.items()}
        if refg not in rbar:
            continue
        # Per-GPU single-pass noise variance, POOLED over rows (mean of the
        # per-row variances -- unbiased for sigma^2, stable over N; a per-row
        # estimate from R~3 passes gives heavy-tailed z). 0 when the repeated
        # passes are bit-identical.
        s2 = {g: float(np.mean(v.var(axis=0, ddof=1))) if R >= 2 else 0.0 for g, v in odds.items()}
        others = [g for g in d["gpus"] if g != refg]
        drift = {g: float(np.mean(np.abs(rbar[g] - rbar[refg]))) for g in others}
        # Arithmetic-noise floor for the deterministic case: the closest GPU's
        # drift (its arithmetic path ~ T4's), never below _ABS_FLOOR.
        floor = max(min(drift.values()), _ABS_FLOOR) if drift else _ABS_FLOOR
        for g in others:
            delta = rbar[g] - rbar[refg]
            mean_abs, max_abs = drift[g], float(np.max(np.abs(delta)))
            flips = int(np.sum((rbar[g] > 0) != (rbar[refg] > 0)))
            sd_stoch = float(np.sqrt(s2[g] / max(R, 1) + s2[refg] / max(R, 1)))
            deterministic = sd_stoch < _DET_TOL
            scale = floor if deterministic else sd_stoch
            z = delta / scale
            rec = dict(
                dataset=d["dataset"], model=d["model"], gpu=g, test="logit_shift", repeats=R,
                mean_abs_logodds_drift=round(mean_abs, 8),
                max_abs_logodds_drift=round(max_abs, 6),
                argmax_flips_vs_ref=flips, deterministic=deterministic,
                noise_scale=round(scale, 8),
                noise_kind="arithmetic_floor" if deterministic else "within_gpu_repeats",
                evalue=logit_drift_evalue(z),
                evalue_scale=gaussian_scale_evalue(z),
                evalue_bias=gaussian_bias_evalue(z),
            )
            rows.append(rec)
    return _aggregate(pd.DataFrame(rows), alpha, key=["model", "gpu", "test"])


# --------------------------------------------------------------------------- #
# shared aggregation
# --------------------------------------------------------------------------- #
def _aggregate(df: pd.DataFrame, alpha: float, key: list[str]) -> dict:
    if df.empty:
        return {"per_dataset": [], "combined": [], "alpha": alpha}
    combined = (
        df.groupby(key)["evalue"]
        .apply(lambda s: combine_evalues(list(s)))
        .reset_index()
        .rename(columns={"evalue": "evalue_product"})
    )
    combined = combined.sort_values("evalue_product", ascending=False, ignore_index=True)
    for a in sorted({alpha, 0.10}):
        combined[f"ebh_reject@{a}"] = [bool(x) for x in ebh(combined["evalue_product"].to_numpy(), a)]
    return _json_safe(
        {
            "alpha": alpha,
            "n_hypotheses": int(len(combined)),
            "per_dataset": df.sort_values("evalue", ascending=False).to_dict("records"),
            "combined": combined.to_dict("records"),
        }
    )


def _json_safe(obj):
    """Recursively make ``obj`` JSON-portable: numpy scalars -> python,
    +/-inf -> the strings "inf"/"-inf", NaN -> None (so the file parses in
    any JSON reader, not just Python's)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v):
            return None
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _print(title: str, res: dict, top: int = 20) -> None:
    print(f"\n===== {title} =====")
    comb = res.get("combined", [])
    if not comb:
        print("  (no artifacts found)")
        return
    df = pd.DataFrame(comb)
    per = pd.DataFrame(res.get("per_dataset", []))
    if not per.empty and "mean_abs_logodds_drift" in per.columns:
        agg = (
            per.groupby(["model", "gpu"])
            .agg(mean_drift=("mean_abs_logodds_drift", "max"),
                 max_drift=("max_abs_logodds_drift", "max"),
                 flips=("argmax_flips_vs_ref", "max"),
                 deterministic=("deterministic", "all"))
            .reset_index()
        )
        df = df.merge(agg, on=["model", "gpu"], how="left")
    show = df.head(top).copy()
    show["evalue_product"] = show["evalue_product"].map(
        lambda v: "inf" if v == "inf" else f"{float(v):.3g}"
    )
    for c in ("mean_drift", "max_drift"):
        if c in show.columns:
            show[c] = show[c].map(lambda v: f"{float(v):.2e}")
    print(show.to_string(index=False))
    rej = df[[c for c in df.columns if c.startswith("ebh_reject@")]].any(axis=1).sum()
    print(f"\n  {int(rej)}/{len(df)} hypotheses rejected by e-BH (drift detected); "
          f"product e-value >= {1 / res['alpha']:.0f} => significant at alpha={res['alpha']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logit-only", action="store_true")
    ap.add_argument("--metric-only", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--ref", default="T4")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    out_dir = os.path.join(args.results_dir, "evalues")
    os.makedirs(out_dir, exist_ok=True)

    if not args.metric_only:
        logit = logit_level(args.results_dir, args.ref, args.alpha)
        with open(os.path.join(out_dir, "logit_level.json"), "w") as f:
            json.dump(logit, f, indent=2, default=str)
        _print("LOGIT-LEVEL drift (softmax input) vs " + args.ref, logit)

    if not args.logit_only:
        metric = metric_level(args.results_dir, args.ref, args.alpha)
        with open(os.path.join(out_dir, "metric_level.json"), "w") as f:
            json.dump(metric, f, indent=2, default=str)
        _print("METRIC-LEVEL fairness drift vs " + args.ref, metric)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
