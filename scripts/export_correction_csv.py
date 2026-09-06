"""Export the effect of the softmax-input correction vector on per-GPU accuracy
as CSV, for one (dataset, dtype) — default ACS income / bf16.

Source: results/correction_results-<dataset>-<model>-<dtype>.json
(the CorrectionView data in web/app/page.tsx).

Layout:
  - one row per (GPU, model);
  - "baseline" column = uncorrected per-GPU accuracy;
  - one column per acc_weight choice fit for that model
    (w=0.00 is the pure spread-minimising correction, w=1.00 keeps accuracy);
  - each cell is  <per-GPU accuracy> ± <cross-GPU accuracy spread for that
    column>  (baseline.accuracy_spread / corrected[w].accuracy_spread) — the
    same ± is shared by every GPU row in a column, since the spread is a
    property of the column, not the cell.

Usage:
    uv run scripts/export_correction_csv.py [dataset] [dtype]
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET = sys.argv[1] if len(sys.argv) > 1 else "acs_income"
DTYPE = sys.argv[2] if len(sys.argv) > 2 else "bf16"
GPU_ORDER = ["T4", "L4", "A100", "H100", "B200"]
HIDDEN_MODELS = {"tabfm", "xgboost"}


def parse_lenient(text: str):
    return json.loads(re.sub(r"\bNaN\b", "null", re.sub(r"-?\bInfinity\b", "null", text)))


def cell(p: float, spread: float) -> str:
    if not isinstance(p, (int, float)) or not math.isfinite(p):
        return ""
    return f"{p:.4f} ± {spread:.4f}"


def main() -> None:
    pat = str(ROOT / "results" / f"correction_results-{DATASET}-*-{DTYPE}.json")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no correction files matching {pat}")

    results = []
    for f in files:
        m = re.match(
            rf"^correction_results-{re.escape(DATASET)}-([^-]+)-{re.escape(DTYPE)}\.json$",
            Path(f).name,
        )
        if not m or m.group(1) in HIDDEN_MODELS:
            continue
        r = parse_lenient(Path(f).read_text(encoding="utf8"))
        r["model"] = m.group(1)
        results.append(r)
    results.sort(key=lambda r: r["model"])

    weights = sorted({c["acc_weight"] for r in results for c in r["corrected"]})
    wcols = [f"w={w:.2f}" for w in weights]

    out = ROOT / "web" / "public" / f"correction_{DATASET}_{DTYPE}.csv"
    with out.open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["GPU", "Model", "baseline", *wcols])
        gpus_seen = {g for r in results for g in r["gpus"]}
        gpus = [g for g in GPU_ORDER if g in gpus_seen]
        for gpu in gpus:
            for r in results:
                base = r["baseline"]
                by_w = {c["acc_weight"]: c for c in r["corrected"]}
                row = [
                    gpu,
                    r["model"],
                    cell(base["per_gpu_accuracy"].get(gpu), base["accuracy_spread"]),
                ]
                for wt in weights:
                    c = by_w.get(wt)
                    row.append(
                        cell(c["per_gpu_accuracy"].get(gpu), c["accuracy_spread"])
                        if c
                        else ""
                    )
                w.writerow(row)
    print(f"wrote {out}  ({len(gpus)} GPUs x {len(results)} models, weights={weights})")


if __name__ == "__main__":
    main()
