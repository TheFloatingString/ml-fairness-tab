# Common commands for the tabular hardware-fairness experiment.
# Run `just` (no args) to list recipes.

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Modal on Windows needs utf-8 stdio (see global CLAUDE.md).
export PYTHONIOENCODING := "utf-8"

# List available recipes.
default:
    @just --list

# Sync the uv-managed environment from uv.lock.
sync:
    uv sync

# Full sweep: GPU x dtype x model. Writes results/results[-<dataset>].json.
# Pass --dataset german_credit, or --refresh-data to rebuild the cache volume.
run *ARGS:
    uv run modal run modal_app.py::main {{ARGS}}

# Full sweep on the German Credit dataset. Writes results/results-german_credit.json.
run-german *ARGS:
    uv run modal run modal_app.py::main --dataset german_credit {{ARGS}}

# Full sweep on the Bank Marketing dataset. Writes results/results-bank_marketing.json.
run-bank *ARGS:
    uv run modal run modal_app.py::main --dataset bank_marketing {{ARGS}}

# Full sweep on the folktables ACSIncome dataset.
run-acs-income *ARGS:
    uv run modal run modal_app.py::main --dataset acs_income {{ARGS}}

# Full sweep on the folktables ACSPublicCoverage dataset.
run-acs-coverage *ARGS:
    uv run modal run modal_app.py::main --dataset acs_coverage {{ARGS}}

# Sweep, TabPFN only.
run-tabpfn:
    uv run modal run modal_app.py::main --tabpfn-only

# Sweep, fp32 only.
run-fp32:
    uv run modal run modal_app.py::main --fp32-only

# Full experiment: sweep TabPFN + TabICL + TabDPT across every dataset x GPU x
# dtype (fp32 + fp16 + bf16, the `main` defaults), then run the correction-vector
# intervention for each of those models on each dataset. Writes results/*.json.
# Use `just run` for the full 6-model set incl. the XGBoost control baseline.
run-quick:
    uv run modal run modal_app.py::main --dataset adult --models tabpfn,tabicl,tabdpt,xgboost
    uv run modal run modal_app.py::main --dataset german_credit --models tabpfn,tabicl,tabdpt,xgboost
    uv run modal run modal_app.py::main --dataset bank_marketing --models tabpfn,tabicl,tabdpt,xgboost
    uv run modal run modal_app.py::main --dataset acs_income --models tabpfn,tabicl,tabdpt,xgboost
    uv run modal run modal_app.py::main --dataset acs_coverage --models tabpfn,tabicl,tabdpt,xgboost
    uv run modal run modal_app.py::correction --dataset adult --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset adult --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset adult --model tabdpt --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset german_credit --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset german_credit --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset german_credit --model tabdpt --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset bank_marketing --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset bank_marketing --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset bank_marketing --model tabdpt --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_income --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_income --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_income --model tabdpt --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_coverage --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_coverage --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_coverage --model tabdpt --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::drift_correlation --models tabpfn,tabicl,tabdpt

# Cross-hardware logit-drift vs fairness-shift correlation (T4 reference):
# 5 GPU x 5 dataset grid per model, one Pearson R each. Writes
# results/drift_correlation-<model>.json.
drift-correlation *ARGS:
    uv run modal run modal_app.py::drift_correlation {{ARGS}}

# Correction-vector scheme: fit a shared softmax-input correction for an
# in-context model on a held-out slice of the training rows that trades
# accuracy against cross-GPU accuracy consistency, evaluated on the full test
# set (the dedicated 20% calibration split fits the vector). Writes
# results/correction_results-<dataset>-<model>-<dtype>.json. Pass
# --model {tabpfn,tabicl,tabfm,tabdpt} / --dataset / --dtypes fp32,fp16,bf16
# (or --dtype-name) / --acc-weights / --steps.
run-correction *ARGS:
    uv run modal run modal_app.py::correction {{ARGS}}

# The correction sweep only: tabpfn + tabicl + tabdpt on every dataset, each at
# fp32 + fp16 + bf16 (same grid as the correction half of `run-quick`, without
# re-running the main accuracy sweep). Writes
# results/correction_results-<dataset>-<model>-<dtype>.json.
run-correction-all:
    uv run modal run modal_app.py::correction --dataset adult --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset adult --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset adult --model tabdpt --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset german_credit --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset german_credit --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset german_credit --model tabdpt --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset bank_marketing --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset bank_marketing --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset bank_marketing --model tabdpt --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_income --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_income --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_income --model tabdpt --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_coverage --model tabpfn --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_coverage --model tabicl --dtypes fp32,fp16,bf16
    uv run modal run modal_app.py::correction --dataset acs_coverage --model tabdpt --dtypes fp32,fp16,bf16

# Margin / noise-sensitivity mechanism probe. Writes results/sensitivity_results.json.
sensitivity:
    uv run modal run modal_app.py::sensitivity

# Cross-hardware fairness-variance report. Pass a path for other datasets,
# e.g. `just analyze results/results-german_credit.json`.
analyze PATH="results/results.json":
    uv run python analyze.py {{PATH}}

# Install the results-dashboard (Next.js) dependencies.
web-install:
    cd web; npm install

# Run the results dashboard at http://localhost:3000 (reads ./results live).
web:
    cd web; npm run dev
