"""Pareto plot of the softmax-input correction-vector trade-off:
    x = cross-GPU accuracy spread (max - min per_gpu_accuracy), log scale  -> lower better
    y = mean accuracy across GPUs                                          -> higher better

One connected curve per model (baseline c=0, then each acc_weight), one subplot
per dataset, for a fixed dtype (default bf16). Layout: 2 rows x 3 cols.
Source: results/correction_results-<dataset>-<model>-<dtype>.json.

Usage:
    uv run --with matplotlib scripts/plot_correction_pareto.py [dtype]
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DTYPE = sys.argv[1] if len(sys.argv) > 1 else "bf16"
DATASETS = ["acs_coverage", "acs_income", "adult", "bank_marketing", "german_credit"]
HIDDEN_MODELS = {"tabfm", "xgboost"}
MODEL_COLORS = {
    "tabpfn": "#6ea8fe",
    "tabicl": "#f6c177",
    "tabdpt": "#9ece6a",
    "mlp": "#7dcfff",
}


def parse_lenient(text: str):
    return json.loads(re.sub(r"\bNaN\b", "null", re.sub(r"-?\bInfinity\b", "null", text)))


def load_series(dataset: str):
    """[(model, [(label, spread, mean_acc), ...]), ...] for one dataset."""
    out = []
    pat = str(ROOT / "results" / f"correction_results-{dataset}-*-{DTYPE}.json")
    for f in sorted(glob.glob(pat)):
        m = re.match(
            rf"^correction_results-{re.escape(dataset)}-([^-]+)-{re.escape(DTYPE)}\.json$",
            Path(f).name,
        )
        if not m or m.group(1) in HIDDEN_MODELS:
            continue
        d = parse_lenient(Path(f).read_text(encoding="utf8"))
        pts = [("baseline", d["baseline"]["accuracy_spread"], d["baseline"]["mean_accuracy"])]
        for c in sorted(d["corrected"], key=lambda c: c["acc_weight"]):
            pts.append((f"w={c['acc_weight']:.2f}", c["accuracy_spread"], c["mean_accuracy"]))
        out.append((m.group(1), pts))
    return out


def draw(ax, series) -> None:
    positive = [x for _, pts in series for _, x, _ in pts if x > 0]
    xfloor = (min(positive) * 0.5) if positive else 1e-5
    for model, pts in series:
        color = MODEL_COLORS.get(model, "#8a8f9c")
        spts = sorted(
            ((label, max(x, xfloor), y) for label, x, y in pts), key=lambda p: p[1]
        )
        ax.plot([p[1] for p in spts], [p[2] for p in spts], "-",
                color=color, alpha=0.45, zorder=1)
        for label, x, y in spts:
            base = label == "baseline"
            ax.scatter(x, y, s=150 if base else 34, color=color,
                       marker="*" if base else "o",
                       edgecolor="#222" if base else "none",
                       linewidth=0.5, zorder=4 if base else 3,
                       label=model if (not base and label == "w=0.00") else None)
            if label in ("baseline", "w=0.00", "w=1.00"):
                ax.annotate(label, (x, y), textcoords="offset points",
                            xytext=(4, 3), fontsize=6)
    ax.set_xscale("log")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#c8c8c8", linewidth=0.6)
    ax.grid(True, which="minor", color="#ececec", linewidth=0.4)
    ax.set_axisbelow(True)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    flat = axes.flatten()

    for ax, dataset in zip(flat, DATASETS):
        series = load_series(dataset)
        if not series:
            ax.set_visible(False)
            continue
        draw(ax, series)
        ax.set_title(
            dataset.replace("_", " ").title().replace("Acs", "ACS")
        )
        ax.set_xlabel("Cross-GPU accuracy spread (log)   ← better")
        ax.set_ylabel("Mean accuracy across GPUs   → better")
        ax.legend(title="model", frameon=False, fontsize=8)

    for ax in flat[len(DATASETS):]:
        ax.set_visible(False)

    fig.suptitle(
        "Correction-Vector Pareto: Accuracy vs Cross-GPU Spread",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = ROOT / "web" / "public" / f"correction_pareto_all_{DTYPE}.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
