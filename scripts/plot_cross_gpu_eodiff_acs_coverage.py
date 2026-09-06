"""Export a matplotlib PNG of the dashboard chart
"Between-GPU spread within each float precision" for the ACS coverage dataset,
metric = equal opportunity diff, one subplot per sensitive attribute
(age, gender, race).

Mirrors web/app/page.tsx (CrossGpuByPrecision):
  - for each float precision, bar height = range (max - min) of the metric
    across GPU types, with model / dataset / dtype fixed (needs >=2 GPU runs);
  - error bar = sqrt(SE_hi^2 + SE_lo^2) over the two extreme GPU configs,
    using the analytic diff_se from tabular_exp/fairness.py.

Usage:
    uv run --with matplotlib --with numpy \
        scripts/plot_cross_gpu_eodiff_acs_coverage.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT = (
    Path(__file__).resolve().parent.parent
    / "web"
    / "public"
    / "cross_gpu_eodiff_acs_coverage.png"
)

DATASET = "acs_coverage"
DATASET_FILE = "results-acs_coverage.json"
METRIC_LABEL = "Equal opportunity diff"
CRITERION = "equal_opportunity"
ATTRS = ["age", "gender", "race"]
DTYPE_ORDER = ["fp32", "fp16", "bf16"]
HIDDEN_MODELS = {"tabfm", "xgboost"}
MODEL_COLORS = {
    "tabpfn": "#6ea8fe",
    "tabicl": "#f6c177",
    "tabdpt": "#9ece6a",
    "tabfm": "#bb9af7",
    "mlp": "#7dcfff",
    "xgboost": "#f7768e",
}


def parse_lenient(text: str):
    return json.loads(re.sub(r"\bNaN\b", "null", re.sub(r"-?\bInfinity\b", "null", text)))


def metric_value(row: dict, attr: str) -> float:
    blk = (row.get("by_attribute") or {}).get(attr)
    return blk[CRITERION]["diff"] if blk else math.nan


def metric_se(row: dict, attr: str) -> float:
    blk = (row.get("by_attribute") or {}).get(attr)
    se = blk[CRITERION].get("diff_se") if blk else None
    return se if isinstance(se, (int, float)) else math.nan


def spread_se(pairs: list[tuple[float, float]]) -> float:
    clean = [(v, se) for v, se in pairs if math.isfinite(v)]
    if len(clean) < 2:
        return math.nan
    hi = max(clean, key=lambda t: t[0])
    lo = min(clean, key=lambda t: t[0])
    sh = hi[1] if math.isfinite(hi[1]) else 0.0
    sl = lo[1] if math.isfinite(lo[1]) else 0.0
    return math.sqrt(sh * sh + sl * sl)


def main() -> None:
    rows = parse_lenient((RESULTS_DIR / DATASET_FILE).read_text(encoding="utf8"))
    rows = [r for r in rows if r["model"] not in HIDDEN_MODELS]
    models = sorted({r["model"] for r in rows})
    dtypes = sorted({r["dtype"] for r in rows}, key=DTYPE_ORDER.index)

    fig, axes = plt.subplots(
        1, len(ATTRS), figsize=(22, 7), sharey=False, constrained_layout=True
    )
    x = np.arange(len(dtypes))
    n = len(models)
    width = 0.8 / n

    for ax, attr in zip(axes, ATTRS):
        top = 0.0
        for i, model in enumerate(models):
            offs = (i - (n - 1) / 2) * width
            heights, errs = [], []
            for dt in dtypes:
                pairs = [
                    (metric_value(r, attr), metric_se(r, attr))
                    for r in rows
                    if r["model"] == model and r["dtype"] == dt
                ]
                pairs = [p for p in pairs if math.isfinite(p[0])]
                if len(pairs) > 1:
                    vs = [v for v, _ in pairs]
                    heights.append(max(vs) - min(vs))
                    errs.append(spread_se(pairs))
                else:
                    heights.append(math.nan)
                    errs.append(math.nan)
            bars = ax.bar(
                x + offs,
                heights,
                width,
                label=model,
                color=MODEL_COLORS.get(model, "#8a8f9c"),
                yerr=errs,
                capsize=3,
                error_kw={"elinewidth": 1.2, "ecolor": "#333"},
            )
            for b, h, e in zip(bars, heights, errs):
                if not math.isfinite(h):
                    continue
                top = max(top, h + (e if math.isfinite(e) else 0))
                cap = f"{h:.4f}\n±{e:.4f}" if math.isfinite(e) else f"{h:.4f}"
                ax.annotate(
                    cap,
                    (b.get_x() + b.get_width() / 2, h + (e if math.isfinite(e) else 0)),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_title(attr.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(dtypes)
        ax.set_xlabel("Float precision")
        ax.set_ylim(0, top * 1.22 if top > 0 else 1)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.grid(True, which="major", color="#c8c8c8", linewidth=0.7)
        ax.yaxis.grid(True, which="minor", color="#e2e2e2", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel("EO-diff spread  (max − min across GPUs)")

    axes[0].legend(title="model", frameon=False)
    fig.suptitle(
        "Between-GPU Spread Within Each Float Precision — "
        "ACS Coverage, Equal Opportunity Diff",
        fontsize=13,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
