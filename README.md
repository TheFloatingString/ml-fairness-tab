# tabular-exp

Measuring **hardware-induced fairness variance** in frozen tabular models: how
much do per-subgroup accuracy and fairness disparities move when only the GPU
type or compute dtype (fp32/fp16/bf16) changes, with weights, data, and seed
fixed? Ported from arXiv:2312.03886 to TabPFN / TabICL / TabDPT / MLP / XGBoost
(XGBoost is the near-zero control). Datasets: adult, german_credit,
bank_marketing, acs_income, acs_coverage.

All GPU work runs on [Modal](https://modal.com); the local machine only needs
the `modal` client. Uses `uv` and `just`.

## Setup

```
uv sync
```

Modal on Windows needs `PYTHONIOENCODING=utf-8` (the `.justfile` sets this).

## Common commands

| Command | Description |
| --- | --- |
| `just run [--dataset X] [--refresh-data]` | Full GPU × dtype × model sweep → `results/results[-<dataset>].json` |
| `just run-quick` | End-to-end experiment: logit-drift screen → sweep + correction-vector intervention → e-value analysis. Tees logs to `results/logs/`. |
| `just evalues` | E-value tests for cross-hardware drift over existing `results/` artifacts (no Modal) |
| `just drift-correlation` | Logit-drift vs fairness-shift Pearson correlation (T4 reference) |
| `just run-correction` | Fit a shared softmax-input correction trading accuracy for cross-GPU consistency |
| `just sensitivity` | Margin / noise-sensitivity mechanism probe |
| `just analyze [path]` | Cross-hardware fairness-variance report |
| `just web` | Results dashboard (Next.js) at http://localhost:3000 |

## Layout

- `modal_app.py` — Modal pipeline (inference, correction, drift, sensitivity)
- `run_quick.py` — sequential driver for the full sweep
- `analyze.py`, `evalue_analysis.py` — offline analysis
- `tabular_exp/` — datasets, models, fairness metrics, e-values, correction
- `scripts/` — CSV exports and plots
- `web/` — Next.js results dashboard (reads `./results` live)
- `results/` — generated JSON artifacts
