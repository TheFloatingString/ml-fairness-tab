"""Export the "Softmax-input drift between GPUs (e-value test)" heat grid for a
single dataset (default: bank_marketing) as CSV, laid out exactly as the
dashboard's EvalueLogitView renders it:

  - one row per model, one column per GPU (GPU_ORDER);
  - each cell is log10 of the per-dataset e-value, formatted like the web viewer
    ("+4.3", "-0.1", "∞", "−∞"); empty ("·") where the grid shows no data
    (the T4 column — T4 is the reference GPU).

Usage:
    uv run scripts/export_logit_drift_csv.py [dataset]
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "evalues" / "logit_level.json"

DATASET = sys.argv[1] if len(sys.argv) > 1 else "bank_marketing"
GPU_ORDER = ["T4", "L4", "A100", "H100", "B200"]
ROW_LABEL = "log10 E"  # the dashboard's corner cell ("log₁₀ E")


def ev_num(v) -> float:
    return math.inf if v == "inf" else float(v)


def fmt_log10(v) -> str:
    """Port of fmtLog10() in web/app/page.tsx."""
    n = ev_num(v)
    if not math.isfinite(n):
        return "∞" if n > 0 else "−∞"
    if n <= 0:
        return "−∞"
    l = math.log10(n)
    return ("+" if l >= 0 else "") + f"{l:.1f}"


def main() -> None:
    level = json.loads(SRC.read_text(encoding="utf8"))
    per_ds = [r for r in level["per_dataset"] if r["dataset"] == DATASET]
    if not per_ds:
        sys.exit(f"no per_dataset rows for dataset {DATASET!r} in {SRC}")

    models = sorted({r["model"] for r in level["combined"]})
    gpus = sorted(
        {r["gpu"] for r in level["combined"]},
        key=lambda g: GPU_ORDER.index(g) if g in GPU_ORDER else 99,
    )
    lookup = {(r["model"], r["gpu"]): r for r in per_ds}

    out = ROOT / "web" / "public" / f"logit_drift_{DATASET}.csv"
    with out.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow([ROW_LABEL, *gpus])
        for m in models:
            row = [m]
            for g in gpus:
                r = lookup.get((m, g))
                row.append(fmt_log10(r["evalue"]) if r else "·")
            w.writerow(row)
    print(f"wrote {out}  ({len(models)} rows x {len(gpus)} GPU cols)")


if __name__ == "__main__":
    main()
