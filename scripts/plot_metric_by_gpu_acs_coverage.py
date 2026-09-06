"""Export a matplotlib PNG of the dashboard chart
"Metric per float type, grouped by GPU" for:
  dataset = ACS coverage, model = tabicl,
  metric  = equal opportunity diff, sensitive attribute = age.

Mirrors web/app/page.tsx (ConfigBreakdown):
  - bars grouped by float type, one bar per GPU inside each group;
  - bar height = the raw metric value for that (GPU, dtype) config;
  - error bar = +/-1 SE (analytic diff_se from tabular_exp/fairness.py);
  - y-axis starts at 0.

Usage:
    uv run --with matplotlib --with numpy \
        scripts/plot_metric_by_gpu_acs_coverage.py
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
    / "metric_by_gpu_acs_coverage_tabicl_eodiff_age.png"
)

DATASET_FILE = "results-acs_coverage.json"
MODEL = "tabicl"
CRITERION = "equal_opportunity"
ATTR = "age"
DTYPE_ORDER = ["fp32", "fp16", "bf16"]
GPU_ORDER = ["T4", "L4", "A100", "H100", "B200"]
GPU_COLORS = {
    "T4": "#6ea8fe",
    "L4": "#f6c177",
    "A100": "#9ece6a",
    "H100": "#bb9af7",
    "B200": "#f7768e",
}


def parse_lenient(text: str):
    return json.loads(re.sub(r"\bNaN\b", "null", re.sub(r"-?\bInfinity\b", "null", text)))


def metric_value(row: dict) -> float:
    blk = (row.get("by_attribute") or {}).get(ATTR)
    return blk[CRITERION]["diff"] if blk else math.nan


def metric_se(row: dict) -> float:
    blk = (row.get("by_attribute") or {}).get(ATTR)
    se = blk[CRITERION].get("diff_se") if blk else None
    return se if isinstance(se, (int, float)) else math.nan


def main() -> None:
    rows = parse_lenient((RESULTS_DIR / DATASET_FILE).read_text(encoding="utf8"))
    rows = [r for r in rows if r["model"] == MODEL]
    dtypes = sorted({r["dtype"] for r in rows}, key=DTYPE_ORDER.index)
    gpus = sorted({r["gpu"] for r in rows}, key=GPU_ORDER.index)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dtypes))
    n = len(gpus)
    width = 0.8 / n
    top = 0.0

    for i, gpu in enumerate(gpus):
        offs = (i - (n - 1) / 2) * width
        heights, errs = [], []
        for dt in dtypes:
            hit = next(
                (r for r in rows if r["gpu"] == gpu and r["dtype"] == dt), None
            )
            v = metric_value(hit) if hit else math.nan
            e = metric_se(hit) if hit else math.nan
            heights.append(v)
            errs.append(e)
            if math.isfinite(v):
                top = max(top, v + (e if math.isfinite(e) else 0))
        bars = ax.bar(
            x + offs,
            heights,
            width,
            label=gpu,
            color=GPU_COLORS.get(gpu, "#8a8f9c"),
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "#333"},
        )
        for b, h, e in zip(bars, heights, errs):
            if not math.isfinite(h):
                continue
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

    ax.set_xticks(x)
    ax.set_xticklabels(dtypes)
    ax.set_xlabel("Float type")
    ax.set_ylabel("Equal opportunity diff (age)")
    ax.set_ylim(0, top * 1.18 if top > 0 else 1)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.grid(True, which="major", color="#c8c8c8", linewidth=0.7)
    ax.yaxis.grid(True, which="minor", color="#e2e2e2", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(title="GPU", frameon=False, ncol=n)
    ax.set_title(
        "Metric Per Float Type, Grouped by GPU — "
        "ACS Coverage, tabICL, Equal Opportunity Diff (Age)"
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
