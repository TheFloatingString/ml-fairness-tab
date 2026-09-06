"""Export a matplotlib PNG of the dashboard chart
"Metric spread across GPU x float type" for metric=accuracy, float type=bf16.

Mirrors the logic in web/app/page.tsx (SpreadOverview):
  - bar height = range (max - min) of accuracy over every GPU run at bf16,
    with model / dataset / dtype held fixed;
  - error bar = sqrt(SE_hi^2 + SE_lo^2), the two extreme configs, where
    SE is the Wald accuracy SE sqrt(p(1-p)/n), n = sum of subgroup counts.

Each bar is annotated with "spread +/- SE".

Usage:
    python scripts/plot_spread_gpu_bf16.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "web" / "public" / "spread_gpu_bf16_accuracy.png"

DTYPE = "bf16"
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
    """Python's json.dump emits bare NaN/Infinity; map them to null like the app."""
    return json.loads(re.sub(r"\bNaN\b", "null", re.sub(r"-?\bInfinity\b", "null", text)))


def load_sweep() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(RESULTS_DIR.glob("results*.json")):
        m = re.match(r"^results-(.+)\.json$", f.name)
        dataset = m.group(1) if m else "adult"
        rows = parse_lenient(f.read_text(encoding="utf8"))
        out[dataset] = [r for r in rows if r["model"] not in HIDDEN_MODELS]
    return out


def accuracy_se(row: dict) -> float:
    n = sum((row.get("subgroup_count") or {}).values())
    p = row["accuracy"]
    return math.sqrt(max(p * (1 - p), 0) / n) if n > 0 else math.nan


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
    sweep = load_sweep()
    datasets = sorted(sweep)
    models = sorted({r["model"] for rows in sweep.values() for r in rows})

    # values[model][dataset] = (spread, spread_se)
    values: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in models}
    for dataset in datasets:
        for model in models:
            pairs = [
                (r["accuracy"], accuracy_se(r))
                for r in sweep[dataset]
                if r["model"] == model and r["dtype"] == DTYPE
            ]
            pairs = [p for p in pairs if isinstance(p[0], (int, float)) and math.isfinite(p[0])]
            if not pairs:
                continue
            vs = [v for v, _ in pairs]
            values[model][dataset] = (max(vs) - min(vs), spread_se(pairs))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(datasets))
    n = len(models)
    width = 0.8 / n

    for i, model in enumerate(models):
        offs = (i - (n - 1) / 2) * width
        heights = [values[model].get(d, (math.nan, math.nan))[0] for d in datasets]
        errs = [values[model].get(d, (math.nan, math.nan))[1] for d in datasets]
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
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy spread  (max − min)")
    ax.set_title("Metric Spread Across GPU × Float Type — Accuracy, bf16")
    ax.legend(title="model", frameon=False)
    ax.margins(y=0.18)
    ax.set_ylim(bottom=0)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.grid(True, which="major", color="#c8c8c8", linewidth=0.7)
    ax.yaxis.grid(True, which="minor", color="#e2e2e2", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
