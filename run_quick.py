"""Sequential runner for the `run-quick` ablation sweep (invoked by
`just run-quick`).

Owns the list of Modal commands that used to be inlined in the `.justfile`
recipe. Each step's combined stdout+stderr is streamed to the console *and*
appended to `results/logs/run-quick-<UTC timestamp>/`:

  - `NN-<entry>-<slug>.log` -- one file per step;
  - `run-quick.log`          -- everything concatenated;
  - `summary.txt`            -- the final step/duration/exit-code table.

A failing step is recorded and the sweep continues to the next step rather
than aborting (so one broken GPU config doesn't cost the whole grid). The
process exits non-zero if any step failed.

No `modal` import here -- this only shells out to `uv run modal run ...`.
"""

import datetime as _dt
import os
import subprocess
import sys
import time

MODELS_MAIN = "tabpfn,tabicl,tabdpt,xgboost"
CORRECTION_MODELS = ["tabpfn", "tabicl", "tabdpt"]
# In-context models screened for softmax-input (logit-level) drift.
INCONTEXT_MODELS = ["tabpfn", "tabicl", "tabdpt"]
DATASETS = ["adult", "german_credit", "bank_marketing", "acs_income", "acs_coverage"]
DTYPES = "fp32,fp16,bf16"

# Independent fp32 forward passes per (in-context model, GPU) captured by the
# `main` step, to calibrate the within-GPU nondeterminism floor for the
# softmax-input (logit-level) drift e-tests. 1 = piggyback on the sweep pass
# (no noise floor); >=2 adds that many-minus-one extra fp32 passes. Raise for
# a tighter noise estimate at linear GPU cost.
LOGIT_REPEATS = 3

RESULTS_DIR = "results"


def _steps() -> list[tuple[str, list[str], bool]]:
    """(slug, argv, local) for every command in the sweep, in run order.
    ``local`` runs ``uv run python <argv>`` instead of ``uv run modal run``."""
    steps: list[tuple[str, list[str], bool]] = []

    # 1. Cheap screening pass first: fp32 in-context logit capture only, then
    #    the logit-level (softmax-input) drift e-test -- gates/pre-empts the
    #    full sweep below.
    for ds in DATASETS:
        steps.append(
            (
                f"logits-{ds}",
                [
                    "modal_app.py::main",
                    "--dataset", ds,
                    "--models", ",".join(INCONTEXT_MODELS),
                    "--collect-logits",
                    "--logit-repeats", str(LOGIT_REPEATS),
                    "--logits-only",
                ],
                False,
            )
        )
    steps.append(("logit-screen", ["evalue_analysis.py", "--logit-only"], True))

    # 2. Full hardware/precision sweep (logits already captured above).
    for ds in DATASETS:
        steps.append(
            (
                f"main-{ds}",
                ["modal_app.py::main", "--dataset", ds, "--models", MODELS_MAIN],
                False,
            )
        )
    for ds in DATASETS:
        for model in CORRECTION_MODELS:
            steps.append(
                (
                    f"correction-{ds}-{model}",
                    [
                        "modal_app.py::correction",
                        "--dataset", ds,
                        "--model", model,
                        "--dtypes", DTYPES,
                    ],
                    False,
                )
            )
    steps.append(
        (
            "drift_correlation",
            ["modal_app.py::drift_correlation", "--models", ",".join(CORRECTION_MODELS)],
            False,
        )
    )
    # 3. Post-runtime: full e-value tests (metric-level + logit-level) over
    #    everything the sweep wrote.
    steps.append(("evalues", ["evalue_analysis.py"], True))
    return steps


def _run(argv: list[str], log_path: str, combined, local: bool = False) -> int:
    """Run `uv run modal run <argv>` (or `uv run python <argv>` when
    ``local``), teeing output to console, `log_path`, and the shared
    `combined` handle. Returns the exit code."""
    cmd = ["uv", "run", "python", *argv] if local else ["uv", "run", "modal", "run", *argv]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    header = f"\n$ {' '.join(cmd)}\n"
    print(header, end="", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(header)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            combined.write(line)
        return proc.wait()


def main() -> int:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = os.path.join(RESULTS_DIR, "logs", f"run-quick-{stamp}")
    os.makedirs(log_dir, exist_ok=True)
    combined_path = os.path.join(log_dir, "run-quick.log")

    steps = _steps()
    outcomes: list[tuple[str, float, int, str]] = []
    with open(combined_path, "w", encoding="utf-8") as combined:
        for i, (slug, argv, local) in enumerate(steps, 1):
            log_path = os.path.join(log_dir, f"{i:02d}-{slug}.log")
            print(f"\n===== [{i}/{len(steps)}] {slug} =====", flush=True)
            combined.write(f"\n===== [{i}/{len(steps)}] {slug} =====\n")
            start = time.time()
            try:
                code = _run(argv, log_path, combined, local)
            except Exception as exc:  # noqa: BLE001 -- record and keep going
                combined.write(f"runner error: {exc!r}\n")
                print(f"runner error: {exc!r}", flush=True)
                code = 1
            dur = time.time() - start
            outcomes.append((slug, dur, code, os.path.relpath(log_path)))
            print(f"----- {slug}: exit {code} in {dur:.0f}s -----", flush=True)

    lines = ["step                                  dur     exit  log"]
    for slug, dur, code, path in outcomes:
        lines.append(f"{slug:<36}  {dur:6.0f}s  {code:>4}  {path}")
    failed = [s for s, _, c, _ in outcomes if c != 0]
    lines.append("")
    lines.append(f"{len(outcomes) - len(failed)}/{len(outcomes)} steps ok"
                 + (f"; failed: {', '.join(failed)}" if failed else ""))
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(log_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
