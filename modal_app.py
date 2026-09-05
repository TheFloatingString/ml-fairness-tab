"""Modal pipeline: run frozen-weight inference for TabPFN / TabICL / MLP /
XGBoost across GPU types and compute dtypes, to measure hardware-induced
fairness variance (following arXiv:2312.03886, ported to tabular models).

Everything that touches torch/tabpfn/xgboost/sklearn runs inside Modal
functions. The local_entrypoint only triggers remote calls and reads back
small JSON metric dicts, so the local machine only needs the `modal` client
(plus `pandas` for analyze.py's offline report).

Speed notes:
  - The TabPFN checkpoint is baked into the image (`_cache_tabpfn_weights`),
    so cold containers don't re-download it.
  - The prepared dataset + trained baselines are cached on a Modal Volume
    (`data_volume`); `prepare_data` populates it once and every inference
    container reads it from `/cache` instead of receiving it as an argument.
  - `main()` spawns all (model, dtype, gpu) configs concurrently.

Datasets (`--dataset`): "adult" (default), "german_credit", "bank_marketing",
"acs_income", "acs_coverage" (the last two via folktables, one state-year of
ACS PUMS). Each is cached on the Volume separately; results go to
results/results[-<dataset>].json.

Usage:
    export PYTHONIOENCODING="utf-8"; modal run modal_app.py::main
    export PYTHONIOENCODING="utf-8"; modal run modal_app.py::main --tabpfn-only --fp32-only
    export PYTHONIOENCODING="utf-8"; modal run modal_app.py::main --dataset german_credit
    export PYTHONIOENCODING="utf-8"; modal run modal_app.py::main --tabpfn-only --dtypes fp16,bf16
    export PYTHONIOENCODING="utf-8"; modal run modal_app.py::main --refresh-data

`--models` (comma-separated subset of tabpfn,tabicl,tabfm,tabdpt,mlp,xgboost) and
`--dtypes` (subset of fp32,fp16,bf16) override the defaults; xgboost always
stays fp32 and runs once (no GPU-dependent kernel path).

Note: `Function.with_options(gpu=...)` for dynamic per-call GPU selection
is a newer Modal feature — pin/check `modal` version if this errors, older
versions require one @app.function per GPU type instead.
"""

import os

import modal

app = modal.App("tabular-hardware-fairness")


def _cache_model_weights():
    """Runs at image-build time so the TabPFN, TabICL, TabFM and TabDPT
    checkpoints are downloaded once into an image layer instead of on every
    cold container start."""
    import numpy as np
    from tabicl import TabICLClassifier
    from tabpfn import TabPFNClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((32, 5)).astype(np.float32)
    y = (rng.random(32) > 0.5).astype(int)
    X_new = rng.standard_normal((4, 5)).astype(np.float32)

    for cls in (TabPFNClassifier, TabICLClassifier):
        clf = cls(device="cpu")
        clf.fit(X, y)
        clf.predict(X_new)

    # Google TabFM: weights auto-download from HF (google/tabfm-1.0.0-pytorch)
    # on first load — pull them into an image layer here.
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch

    tabfm_model = tabfm_v1_0_0_pytorch.load(model_type="classification", device="cpu")
    tabfm_clf = TabFMClassifier(model=tabfm_model)
    tabfm_clf.fit(X, y)
    tabfm_clf.predict(X_new)

    # Layer6 TabDPT: weights auto-download from HF (Layer6/TabDPT) on first
    # construction — bake them in here.
    from tabdpt import TabDPTClassifier

    tabdpt_clf = TabDPTClassifier(device="cpu", use_flash=False, compile=False, verbose=False)
    tabdpt_clf.fit(X, y)
    tabdpt_clf.predict(X_new, n_ensembles=1, seed=0)


image = (
    modal.Image.debian_slim(python_version="3.12")
    # `git` for the `tabfm @ git+https://...` install; `g++` for building
    # TabDPT's native deps (faiss).
    .apt_install("git", "g++")
    .pip_install(
        "torch",
        "tabpfn",
        "tabicl",
        "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git",
        "tabdpt",
        "xgboost",
        "scikit-learn",
        "scipy",
        "fairlearn",
        "folktables",
        "pandas",
        "numpy",
    )
    # TabPFN v3 (default): non-commercial license (tabpfn-3-license-v1.0),
    # unresolved for paper use as of 2026-08-08 — local testing/exploration
    # only, per explicit confirmation. Needs the tabpfn-token secret below
    # for gated license acceptance.
    # TabFM (google/tabfm-1.0.0-pytorch): TabFM Non-Commercial License v1.0 —
    # same local-only exploration constraint. Installed from git (the PyPI
    # `tabfm` ships a broken loader); weights pulled by _cache_model_weights.
    # TabDPT (Layer6/TabDPT): permissive license; weights pulled by
    # _cache_model_weights.
    .run_function(_cache_model_weights, secrets=[modal.Secret.from_name("tabpfn-token")])
    # Persist folktables' raw ACS PUMS download on the data Volume.
    .env({"FOLKTABLES_ROOT": "/cache/folktables"})
    .add_local_python_source("tabular_exp")
)

data_volume = modal.Volume.from_name("tabular-exp-data", create_if_missing=True)
CACHE_DIR = "/cache"
DEFAULT_DATASET = "adult"
# All local JSON artifacts (sweep / correction / sensitivity) land here.
RESULTS_DIR = "results"


def _prepared_path(dataset: str) -> str:
    return f"{CACHE_DIR}/prepared-{dataset}.json"


def _out_path(stem: str, dataset: str) -> str:
    """`results/results.json` for the default dataset,
    `results/results-<dataset>.json` otherwise."""
    name = f"{stem}.json" if dataset == DEFAULT_DATASET else f"{stem}-{dataset}.json"
    return os.path.join(RESULTS_DIR, name)


def _write_json(path: str, obj) -> None:
    """Dump `obj` as indented JSON, creating the parent directory if needed."""
    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote {path}")


def _write_prediction_log(name: str, obj) -> None:
    """Write a normalized per-prediction log to `results/predictions/<name>`.

    Shape: `{dataset, entrypoint, n_test, y_true[], sensitive{attr: [...]},
    predictions[{...config, correct[]}]}` -- sensitive labels and `y_true` are
    stored once; each config contributes an index-aligned `correct` vector
    (1 = correct, 0 = incorrect). Lets post-hoc fairness analysis run without
    re-executing the Modal pipeline."""
    _write_json(os.path.join(RESULTS_DIR, "predictions", name), obj)


DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
# CPU control ("None") temporarily dropped to speed up sweeps — restore by
# prepending `None` here (and see the xgboost branch in main()).
GPU_TYPES = ["T4", "L4", "A100", "H100", "B200"]


def _load_prepared(dataset: str) -> dict:
    import json

    with open(_prepared_path(dataset)) as f:
        return json.load(f)


@app.function(image=image, timeout=120, volumes={CACHE_DIR: data_volume})
def test_labels(dataset: str = DEFAULT_DATASET) -> dict:
    """Ground-truth labels and per-row sensitive-attribute labels for the
    test split, read straight from the prepared cache. Fetched once per
    dataset so the shared block of a prediction log isn't shipped per config
    (`sensitive_test` is already stored as `{attr: [str, ...]}`)."""
    data = _load_prepared(dataset)
    return {"y_test": data["y_test"], "sensitive_test": data["sensitive_test"]}


@app.function(image=image, timeout=600, volumes={CACHE_DIR: data_volume})
def prepare_data(dataset: str = DEFAULT_DATASET, train_baselines: bool = True, refresh: bool = False) -> dict:
    """Loads the dataset split and trains MLP/XGBoost once (CPU, fixed seed),
    caching the result as JSON on `data_volume` (one file per dataset) so
    later runs skip this entirely. `train_baselines=False` skips MLP/XGBoost;
    `refresh=True` forces a rebuild. Returns small metadata only.

    Splits are 60 / 20 / 20 train / calib / test (seed 42); the cached JSON
    holds the full calib + test slices, so for the large ACS datasets this
    file runs to a few hundred MB on the Volume."""
    import json
    import os

    from tabular_exp.datasets import CONTEXT_CAP

    path = _prepared_path(dataset)
    # "X_calib" marks the 60/20/20 schema; a mismatched in-context size means
    # CONTEXT_CAP changed. Either way the cache is stale and must be rebuilt
    # regardless of the refresh flag.
    if not refresh and os.path.exists(path):
        cached = _load_prepared(dataset)
        fresh_schema = "X_calib" in cached
        cap_ok = cached.get("split_sizes", {}).get("context") == min(
            CONTEXT_CAP, cached.get("split_sizes", {}).get("train_full", CONTEXT_CAP)
        )
        if fresh_schema and cap_ok:
            have_baselines = cached.get("mlp_state") is not None
            if have_baselines or not train_baselines:
                return {
                    "cached": True,
                    "has_baselines": have_baselines,
                    "n_features": cached["n_features"],
                }

    import numpy as np

    from tabular_exp.datasets import load

    split = load(dataset)

    # XGBoost / MLP train on the full 80% (train + calib), uncapped; the
    # tabular foundation models get the <= CONTEXT_CAP in-context set.
    X_base = np.concatenate([split.X_train_full, split.X_calib])
    y_base = np.concatenate([split.y_train_full, split.y_calib])

    mlp_state = xgb_raw = None
    if train_baselines:
        from tabular_exp.models import train_mlp, train_xgboost

        mlp_state = train_mlp(X_base, y_base)
        xgb_raw = bytes(train_xgboost(X_base, y_base)).hex()

    def sens(df):
        return {c: df[c].astype(str).tolist() for c in df.columns}

    prepared = {
        "X_train": split.X_train.tolist(),
        "y_train": split.y_train.tolist(),
        "X_calib": split.X_calib.tolist(),
        "y_calib": split.y_calib.tolist(),
        "X_test": split.X_test.tolist(),
        "y_test": split.y_test.tolist(),
        "sensitive_train": sens(split.sensitive_train),
        "sensitive_calib": sens(split.sensitive_calib),
        "sensitive_test": sens(split.sensitive_test),
        "mlp_state": mlp_state.hex() if mlp_state is not None else None,
        "xgb_raw": xgb_raw,
        "n_features": int(split.X_train.shape[1]),
        "split_sizes": {
            "context": int(len(split.X_train)),
            "train_full": int(len(split.X_train_full)),
            "calib": int(len(split.X_calib)),
            "test": int(len(split.X_test)),
        },
    }
    with open(path, "w") as f:
        json.dump(prepared, f)
    data_volume.commit()
    return {"cached": False, "has_baselines": mlp_state is not None, "n_features": prepared["n_features"]}


@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("tabpfn-token")],
    volumes={CACHE_DIR: data_volume},
)
def run_inference(
    dataset: str,
    model_name: str,
    dtype_name: str,
    repeats: int = 1,
    return_logits: bool = False,
) -> dict:
    """One hardware/precision config. Returns aggregate fairness metrics plus
    per-row ``correct`` (1/0) and ``y_pred``.

    ``return_logits`` (in-context models only): also return ``logprob`` --
    ``repeats`` independent forward passes of per-row ``log(predict_proba)``,
    shape ``(repeats, N_test, C)``. Pass 0's argmax is used as ``y_pred`` so
    the hard predictions match the logged softmax inputs exactly (identical to
    ``predict()`` for these models -- ``predict`` is argmax of
    ``predict_proba``). ``repeats > 1`` gives the within-GPU nondeterminism
    floor for the logit-drift e-tests; it costs one extra forward pass each."""
    import numpy as np
    import pandas as pd
    import torch

    from tabular_exp.fairness import correct_vector, summarize
    from tabular_exp.models import (
        predict_mlp,
        predict_proba_incontext,
        predict_tabdpt,
        predict_tabfm,
        predict_tabicl,
        predict_tabpfn,
        predict_xgboost,
    )

    incontext = {"tabpfn", "tabicl", "tabfm", "tabdpt"}

    data = _load_prepared(dataset)
    X_train = np.array(data["X_train"], dtype=np.float32)
    y_train = np.array(data["y_train"])
    X_test = np.array(data["X_test"], dtype=np.float32)
    y_test = np.array(data["y_test"])
    sensitive_test = pd.DataFrame(data["sensitive_test"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, DTYPES[dtype_name])

    logprob_runs = n_classes = None
    if return_logits and model_name in incontext:
        n_passes = max(1, repeats)
        probas = [
            np.asarray(predict_proba_incontext(model_name, X_train, y_train, X_test, device, dtype))
            for _ in range(n_passes)
        ]
        preds = probas[0].argmax(axis=1)
        n_classes = int(probas[0].shape[1])
        logprob_runs = [
            np.round(np.log(np.clip(p, 1e-12, 1.0)), 6).tolist() for p in probas
        ]
    elif model_name == "tabpfn":
        preds = predict_tabpfn(X_train, y_train, X_test, device, dtype)
    elif model_name == "tabicl":
        preds = predict_tabicl(X_train, y_train, X_test, device, dtype)
    elif model_name == "tabfm":
        preds = predict_tabfm(X_train, y_train, X_test, device, dtype)
    elif model_name == "tabdpt":
        preds = predict_tabdpt(X_train, y_train, X_test, device, dtype)
    elif model_name == "mlp":
        preds = predict_mlp(bytes.fromhex(data["mlp_state"]), data["n_features"], X_test, device, dtype)
    elif model_name == "xgboost":
        preds = predict_xgboost(bytes.fromhex(data["xgb_raw"]), X_test)
    else:
        raise ValueError(model_name)

    preds = np.asarray(preds)
    metrics = summarize(y_test, preds, sensitive_test)
    out = {
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "correct": correct_vector(y_test, preds),
        "y_pred": preds.astype(int).tolist(),
        **metrics,
    }
    if logprob_runs is not None:
        out["logprob"] = logprob_runs
        out["n_classes"] = n_classes
    return out


@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("tabpfn-token")],
    gpu="T4",
    volumes={CACHE_DIR: data_volume},
)
def run_sensitivity(dataset: str, sigma_fracs: list[float], n_repeats: int) -> dict:
    """Single-hardware mechanism probe: does a subgroup's smaller decision
    margin predict higher flip-rate under tiny input perturbations? This
    isolates the mechanism from the hardware confound in run_inference."""
    import numpy as np
    import pandas as pd
    import torch

    from tabular_exp.models import _fit_tabpfn
    from tabular_exp.sensitivity import baseline_flip_rate, decision_margins, noise_flip_rates, subgroup_means

    data = _load_prepared(dataset)
    X_train = np.array(data["X_train"], dtype=np.float32)
    y_train = np.array(data["y_train"])
    X_test = np.array(data["X_test"], dtype=np.float32)
    sensitive_test = pd.DataFrame(data["sensitive_test"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = _fit_tabpfn(X_train, y_train, device)

    proba_clean = np.asarray(clf.predict_proba(X_test))
    preds_clean = proba_clean.argmax(axis=1)
    margins = decision_margins(proba_clean)

    def predict_fn(X):
        return np.asarray(clf.predict_proba(X)).argmax(axis=1)

    # Zero-noise control: how often does the model disagree with itself on
    # the *unperturbed* input? This isolates TabPFN's own internal
    # ensembling stochasticity from flips caused by the injected noise below.
    baseline_flips = baseline_flip_rate(X_test, preds_clean, predict_fn, n_repeats)

    flip_rates = noise_flip_rates(
        X_test, preds_clean, predict_fn, X_train, sigma_fracs, n_repeats, seed=42
    )

    # Break every subgroup mean down by each sensitive attribute the dataset has.
    def by_attr(values):
        return {
            col: subgroup_means(values, sensitive_test[col])
            for col in sensitive_test.columns
        }

    return {
        "margin_by_subgroup": by_attr(margins),
        "baseline_flip_rate_by_subgroup": by_attr(baseline_flips),
        "baseline_margin_flip_correlation": float(np.corrcoef(margins, baseline_flips)[0, 1]),
        "flip_rate_by_subgroup": {
            str(sigma): by_attr(rates) for sigma, rates in flip_rates.items()
        },
        "margin_flip_correlation": {
            str(sigma): float(np.corrcoef(margins, rates)[0, 1]) for sigma, rates in flip_rates.items()
        },
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("tabpfn-token")],
    volumes={CACHE_DIR: data_volume},
)
def collect_incontext_logprobs(dataset: str, model_name: str, dtype_name: str) -> dict:
    """Run an in-context model (tabpfn / tabicl / tabfm / tabdpt) on this
    container's GPU and return per-row ``log(predict_proba)`` -- the
    softmax-invariant part of the last-layer logits, which the
    correction-vector scheme shifts and re-softmaxes.

    Same in-context training set for both evals (the 60% train slice, capped
    at CONTEXT_CAP), so the test path is identical to the main sweep:
      - ``test_logprob``: predicted on ``X_test`` (20% test split);
      - ``calib_logprob``: predicted on ``X_calib`` (dedicated 20% calib
        split, disjoint from context and test -- no leakage), for fitting ``c``.
    """
    import numpy as np
    import torch

    from tabular_exp.models import predict_proba_incontext

    data = _load_prepared(dataset)
    X_train = np.array(data["X_train"], dtype=np.float32)
    y_train = np.array(data["y_train"])
    X_calib = np.array(data["X_calib"], dtype=np.float32)
    X_test = np.array(data["X_test"], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, DTYPES[dtype_name])

    def logprob(X_eval):
        proba = predict_proba_incontext(model_name, X_train, y_train, X_eval, device, dtype)
        return np.log(np.clip(proba, 1e-12, 1.0)).tolist()

    return {
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "test_logprob": logprob(X_test),
        "calib_logprob": logprob(X_calib),
    }


@app.function(image=image, timeout=900, volumes={CACHE_DIR: data_volume})
def fit_and_eval_correction(
    dataset: str,
    model_name: str,
    test_logprob_by_gpu: dict,
    calib_logprob_by_gpu: dict,
    acc_weights: list[float],
    steps: int,
    dtype_name: str = "fp32",
) -> dict:
    """Fit one correction vector per ``acc_weight`` on the dedicated 20%
    calibration split's logprobs, then score baseline (c=0) and each
    corrected variant on the full test set. CPU-only -- the optimisation is
    over a length-``n_classes`` vector.

    ``T4`` is the reference GPU: every per-GPU metric also gets a
    ``*_vs_t4`` delta, and the baseline block reports the mean centered-logit
    L2 drift of each GPU from T4."""
    import numpy as np
    import pandas as pd

    from tabular_exp.correction import (
        apply_correction,
        centered_logit_l2_drift,
        fit_correction_vector,
    )
    from tabular_exp.fairness import correct_vector, sensitive_labels, summarize

    data = _load_prepared(dataset)
    y_test = np.array(data["y_test"])
    y_calib = np.array(data["y_calib"])
    sensitive_test = pd.DataFrame(data["sensitive_test"])

    gpus = sorted(test_logprob_by_gpu)
    ref = "T4" if "T4" in gpus else gpus[0]
    test_lp = {g: np.array(test_logprob_by_gpu[g], dtype=np.float64) for g in gpus}
    calib_lp = {g: np.array(calib_logprob_by_gpu[g], dtype=np.float64) for g in gpus}
    n_classes = test_lp[gpus[0]].shape[1]

    # variant key ("baseline" / "acc_weight=<w>") -> {gpu: {"correct", "y_pred"}},
    # drained by the caller into a normalized prediction log.
    pred_log: dict[str, dict[str, dict]] = {}

    def eval_c(c: np.ndarray, variant: str) -> dict:
        per_gpu = {}
        by_gpu = {}
        for g in gpus:
            preds = apply_correction(test_lp[g], c).argmax(axis=1)
            per_gpu[g] = summarize(y_test, preds, sensitive_test)
            by_gpu[g] = {
                "correct": correct_vector(y_test, preds),
                "y_pred": preds.astype(int).tolist(),
            }
        pred_log[variant] = by_gpu
        accs = [per_gpu[g]["accuracy"] for g in gpus]
        dps = [per_gpu[g]["demographic_parity_diff"] for g in gpus]
        acc_by_gpu = {g: per_gpu[g]["accuracy"] for g in gpus}
        dp_by_gpu = {g: per_gpu[g]["demographic_parity_diff"] for g in gpus}
        return {
            "per_gpu_accuracy": acc_by_gpu,
            "per_gpu_demographic_parity_diff": dp_by_gpu,
            "accuracy_vs_t4": {g: acc_by_gpu[g] - acc_by_gpu[ref] for g in gpus},
            "demographic_parity_diff_vs_t4": {
                g: dp_by_gpu[g] - dp_by_gpu[ref] for g in gpus
            },
            "mean_accuracy": float(np.mean(accs)),
            "accuracy_spread": float(np.max(accs) - np.min(accs)),
            "demographic_parity_diff_spread": float(np.max(dps) - np.min(dps)),
        }

    results = {
        "model": model_name,
        "gpus": gpus,
        "reference_gpu": ref,
        "n_calib": int(len(y_calib)),
        "n_test": int(len(y_test)),
        "mean_l2_logit_drift_vs_t4": centered_logit_l2_drift(test_lp, ref=ref),
        "baseline": eval_c(np.zeros(n_classes), "baseline"),
        "corrected": [],
    }
    for w in acc_weights:
        c = fit_correction_vector(calib_lp, y_calib, acc_weight=w, steps=steps)
        # `corrected_softmax_input = softmax_input + c`. softmax is
        # shift-invariant, so report c centred on its own mean; for 2 classes
        # that is a single scalar (`c[0] - c[1]`, the logit-odds shift).
        centred = (c - c.mean()).tolist()
        results["corrected"].append(
            {
                "acc_weight": w,
                "correction_vector": c.tolist(),
                "correction_coeff": float(c[0] - c[1]) if n_classes == 2 else centred,
                **eval_c(c, f"acc_weight={w}"),
            }
        )

    predictions = []
    for variant, by_gpu in pred_log.items():
        is_baseline = variant == "baseline"
        for g, rec in by_gpu.items():
            predictions.append(
                {
                    "model": model_name,
                    "dtype": dtype_name,
                    "gpu": g,
                    "variant": "baseline" if is_baseline else "corrected",
                    "acc_weight": None if is_baseline else float(variant.split("=")[1]),
                    "correct": rec["correct"],
                    "y_pred": rec["y_pred"],
                }
            )
    results["prediction_log"] = {
        "dataset": dataset,
        "entrypoint": "correction",
        "n_test": int(len(y_test)),
        "y_true": y_test.tolist(),
        "sensitive": sensitive_labels(sensitive_test),
        "predictions": predictions,
    }
    return results


@app.local_entrypoint()
def correction(
    dataset: str = DEFAULT_DATASET,
    model: str = "tabpfn",
    dtype_name: str = "fp32",
    dtypes: str = "",
    acc_weights: str = "0.0,0.25,0.5,0.75,1.0",
    steps: int = 1000,
    refresh_data: bool = False,
):
    """Correction-vector scheme: fit a shared softmax-input correction for an
    in-context model (``--model`` = tabpfn / tabicl / tabfm / tabdpt) that
    trades accuracy against cross-GPU accuracy consistency. Collects the
    model's logprobs on every GPU in the sweep, fits one vector per
    ``acc_weight`` on the dedicated 20% calibration split (disjoint from the
    in-context training set and the test set), evaluates on the full test
    set, and writes
    results/correction_results-<dataset>-<model>-<dtype>.json.

    ``--dtypes fp32,fp16,bf16`` runs one correction sweep per precision
    (``--dtype-name`` is the single-precision fallback when ``--dtypes`` is
    unset).

    modal run modal_app.py::correction --dataset german_credit
    modal run modal_app.py::correction --model tabicl --dtypes fp32,fp16,bf16 --dataset german_credit
    """
    weights = [float(w) for w in acc_weights.split(",") if w]
    dtype_names = [d for d in dtypes.split(",") if d] or [dtype_name]
    prepare_data.remote(dataset=dataset, train_baselines=False, refresh=refresh_data)

    for dt in dtype_names:
        _run_correction(dataset, model, dt, weights, steps)


def _run_correction(dataset, model, dtype_name, weights, steps):
    pending = {
        gpu: collect_incontext_logprobs.with_options(gpu=gpu).spawn(
            dataset, model, dtype_name
        )
        for gpu in GPU_TYPES
    }
    test_lp, calib_lp = {}, {}
    for gpu, call in pending.items():
        out = call.get()
        test_lp[gpu] = out["test_logprob"]
        calib_lp[gpu] = out["calib_logprob"]
        print(f"{gpu:6s} logprobs from {out['device_name']}")

    res = fit_and_eval_correction.remote(
        dataset, model, test_lp, calib_lp, weights, steps, dtype_name
    )
    plog = res.pop("prediction_log")

    b = res["baseline"]
    ref = res["reference_gpu"]
    print(
        f"\n{model} on {dataset}  GPUs: {res['gpus']}  (full test set n={res['n_test']}, "
        f"calib n={res['n_calib']}, dtype={dtype_name}, ref={ref})"
    )
    drift = res["mean_l2_logit_drift_vs_t4"]
    print("  mean centered-logit L2 drift vs " + ref + ": " + ", ".join(
        f"{g}={drift[g]:.4f}" for g in res["gpus"]
    ))
    def fmt_coeff(entry):
        cc = entry.get("correction_coeff", 0.0)
        return f"{cc:+.3f}" if isinstance(cc, (int, float)) else str([round(v, 3) for v in cc])

    def vs_t4(entry):
        a = max(abs(v) for v in entry["accuracy_vs_t4"].values())
        d = max(abs(v) for v in entry["demographic_parity_diff_vs_t4"].values())
        return f"max|acc-{ref}|={a:.4f}  max|dp_diff-{ref}|={d:.4f}"

    # `corrected_softmax_input = softmax_input + C`; C is the correction
    # coefficient (a scalar logit-odds shift for 2 classes). acc_weight is the
    # loss weight on the accuracy objective that produced each C.
    print(
        f"{'C=+0.000':>16}  acc_weight=--    mean_acc={b['mean_accuracy']:.4f}  "
        f"acc_spread={b['accuracy_spread']:.4f}  dp_diff_spread={b['demographic_parity_diff_spread']:.4f}  "
        f"{vs_t4(b)}  (baseline)"
    )
    for e in res["corrected"]:
        print(
            f"{'C=' + fmt_coeff(e):>16}  acc_weight={e['acc_weight']:.2f}  "
            f"mean_acc={e['mean_accuracy']:.4f}  acc_spread={e['accuracy_spread']:.4f}  "
            f"dp_diff_spread={e['demographic_parity_diff_spread']:.4f}  {vs_t4(e)}"
        )

    # One file per (dataset, model, dtype) so sweeps don't clobber each other.
    res["dataset"] = dataset
    res["dtype"] = dtype_name
    _write_json(
        os.path.join(RESULTS_DIR, f"correction_results-{dataset}-{model}-{dtype_name}.json"),
        res,
    )
    _write_prediction_log(f"correction-{dataset}-{model}-{dtype_name}.json", plog)


@app.function(image=image, timeout=600, volumes={CACHE_DIR: data_volume})
def drift_points(dataset: str, test_logprob_by_gpu: dict, model: str = "") -> dict:
    """For one dataset: per GPU, the mean centered-logit L2 drift from T4 (X)
    and the demographic-parity-diff gap from T4 (Y), using each GPU's own
    argmax predictions. CPU-only."""
    import numpy as np
    import pandas as pd

    from tabular_exp.correction import centered_logit_l2_drift
    from tabular_exp.fairness import correct_vector, sensitive_labels, summarize

    data = _load_prepared(dataset)
    y_test = np.array(data["y_test"])
    sensitive_test = pd.DataFrame(data["sensitive_test"])

    gpus = sorted(test_logprob_by_gpu)
    ref = "T4" if "T4" in gpus else gpus[0]
    lp = {g: np.array(test_logprob_by_gpu[g], dtype=np.float64) for g in gpus}
    drift = centered_logit_l2_drift(lp, ref=ref)

    preds_by_gpu = {g: lp[g].argmax(axis=1) for g in gpus}
    dp = {
        g: summarize(y_test, preds_by_gpu[g], sensitive_test)["demographic_parity_diff"]
        for g in gpus
    }
    return {
        "reference_gpu": ref,
        "prediction_log": {
            "dataset": dataset,
            "entrypoint": "drift",
            "model": model,
            "n_test": int(len(y_test)),
            "y_true": y_test.tolist(),
            "sensitive": sensitive_labels(sensitive_test),
            "predictions": [
                {
                    "model": model,
                    "dtype": "fp32",
                    "gpu": g,
                    "correct": correct_vector(y_test, preds_by_gpu[g]),
                    "y_pred": preds_by_gpu[g].astype(int).tolist(),
                }
                for g in gpus
            ],
        },
        "points": [
            {
                "gpu": g,
                "dataset": dataset,
                "x": float(drift[g]),
                "y": float(abs(dp[g] - dp[ref])),
            }
            for g in gpus
        ],
    }


@app.function(image=image, timeout=120)
def pearson(xs: list[float], ys: list[float]) -> dict:
    from scipy.stats import pearsonr

    r, p = pearsonr(xs, ys)
    return {"pearson_r": float(r), "p_value": float(p)}


@app.local_entrypoint()
def drift_correlation(models: str = "tabpfn,tabicl,tabdpt", refresh_data: bool = False):
    """Cross-hardware logit-drift vs fairness-shift correlation, with T4 as
    the reference. For each in-context model, over the 5 GPUs x 5 datasets
    grid (25 points): X = mean centered-logit L2 drift from T4,
    Y = |demographic_parity_diff - T4's demographic_parity_diff|. Reports one
    Pearson R per model and writes results/drift_correlation-<model>.json.

    modal run modal_app.py::drift_correlation --models tabpfn
    """
    datasets = ["adult", "german_credit", "bank_marketing", "acs_income", "acs_coverage"]
    for ds in datasets:
        prepare_data.remote(dataset=ds, train_baselines=False, refresh=refresh_data)

    for model in [m for m in models.split(",") if m]:
        points = []
        for ds in datasets:
            pending = {
                gpu: collect_incontext_logprobs.with_options(gpu=gpu).spawn(ds, model, "fp32")
                for gpu in GPU_TYPES
            }
            test_lp = {gpu: call.get()["test_logprob"] for gpu, call in pending.items()}
            out = drift_points.remote(ds, test_lp, model)
            points.extend(out["points"])
            _write_prediction_log(f"drift-{model}-{ds}.json", out["prediction_log"])
            print(f"{model:7s} {ds:14s} done ({len(out['points'])} pts)")

        stats = pearson.remote([p["x"] for p in points], [p["y"] for p in points])
        res = {
            "model": model,
            "reference_gpu": "T4",
            "n_points": len(points),
            **stats,
            "points": points,
        }
        print(
            f"\n{model}: Pearson R = {stats['pearson_r']:+.4f} "
            f"(p={stats['p_value']:.4g}) over {len(points)} points\n"
        )
        _write_json(os.path.join(RESULTS_DIR, f"drift_correlation-{model}.json"), res)


@app.local_entrypoint()
def sensitivity(dataset: str = DEFAULT_DATASET, refresh_data: bool = False):
    """Runs the margin/noise-sensitivity mechanism probe (single GPU, no
    hardware sweep) and writes sensitivity_results[-<dataset>].json.

    Usage: modal run modal_app.py::sensitivity --dataset german_credit
    """
    prepare_data.remote(dataset=dataset, train_baselines=False, refresh=refresh_data)
    out = run_sensitivity.remote(dataset, sigma_fracs=[1e-4, 1e-3, 1e-2], n_repeats=10)

    print(f"Ran on: {out['device_name']}")
    print("Mean decision margin by subgroup:", out["margin_by_subgroup"])
    print(
        "Zero-noise (self-disagreement) baseline flip rate:",
        out["baseline_flip_rate_by_subgroup"],
        f"  margin-flip corr={out['baseline_margin_flip_correlation']:.3f}",
    )
    for sigma, rates in out["flip_rate_by_subgroup"].items():
        corr = out["margin_flip_correlation"][sigma]
        print(f"sigma={sigma}: flip_rate={rates}  margin-flip corr={corr:.3f}")

    _write_json(_out_path("sensitivity_results", dataset), out)


INCONTEXT_MODELS = ("tabpfn", "tabicl", "tabfm", "tabdpt")


@app.local_entrypoint()
def main(
    dataset: str = DEFAULT_DATASET,
    tabpfn_only: bool = False,
    fp32_only: bool = False,
    models: str = "",
    dtypes: str = "",
    refresh_data: bool = False,
    collect_logits: bool = False,
    logit_repeats: int = 1,
    logits_only: bool = False,
):
    """Hardware/precision sweep. Writes results/results[-<dataset>].json and
    the per-prediction log results/predictions/predictions-<dataset>.json
    (per-row ``correct`` / ``y_pred`` + sensitive labels, per config).

    ``--collect-logits`` additionally captures, for the in-context models at
    **fp32** (precision held fixed so only hardware varies), ``logit_repeats``
    independent forward passes of per-row ``log(predict_proba)`` on every GPU,
    to results/logits/logits-<dataset>-<model>-fp32.json -- the input to the
    softmax-input (logit-level) drift e-tests. ``--logit-repeats N`` (N >= 2)
    gives the within-GPU nondeterminism floor those tests calibrate against;
    it costs N-1 extra fp32 forward passes per (in-context model, GPU).

    ``--logits-only`` runs *just* that fp32 in-context logit capture (no
    hard-prediction sweep, no results.json / prediction log) -- the cheap
    screening pass that `just run-quick` runs first, so the logit-level
    e-test can gate the full sweep."""
    ALL_MODELS = ["tabpfn", "tabicl", "tabfm", "tabdpt", "mlp", "xgboost"]
    if models:
        models = [m for m in models.split(",") if m]
    elif tabpfn_only:
        models = ["tabpfn"]
    else:
        models = ALL_MODELS
    if logits_only:
        collect_logits = True
        models = [m for m in models if m in INCONTEXT_MODELS]
        fp32_only = True
    dtype_override = [d for d in dtypes.split(",") if d] or None
    need_baselines = (not logits_only) and any(m in ("mlp", "xgboost") for m in models)

    # Populate the dataset cache once (blocks until committed), then fan out.
    prepare_data.remote(dataset=dataset, train_baselines=need_baselines, refresh=refresh_data)
    labels = test_labels.remote(dataset)

    pending = []
    for model_name in models:
        if dtype_override is not None:
            dtype_names = ["fp32"] if model_name == "xgboost" else dtype_override
        elif fp32_only:
            dtype_names = ["fp32"]
        else:
            dtype_names = ["fp32"] if model_name == "xgboost" else ["fp32", "fp16", "bf16"]
        for dtype_name in dtype_names:
            for gpu in GPU_TYPES:
                if model_name == "xgboost" and gpu != GPU_TYPES[0]:
                    continue  # deterministic tree traversal: run once as the invariant control
                want_logits = (
                    collect_logits and model_name in INCONTEXT_MODELS and dtype_name == "fp32"
                )
                call = run_inference.with_options(gpu=gpu).spawn(
                    dataset,
                    model_name,
                    dtype_name,
                    logit_repeats if want_logits else 1,
                    want_logits,
                )
                pending.append(
                    ({"model": model_name, "gpu": gpu, "dtype": dtype_name, "logits": want_logits}, call)
                )

    results = []
    pred_entries = []
    # model -> {"gpus": [...], "device_name": {gpu: name}, "logprob": {gpu: (R,N,C)}}
    logit_bundles: dict = {}
    for meta, call in pending:
        out = call.get()
        correct = out.pop("correct")
        y_pred = out.pop("y_pred")
        logprob = out.pop("logprob", None)
        n_classes = out.pop("n_classes", None)
        results.append({**{k: v for k, v in meta.items() if k != "logits"}, **out})
        pred_entries.append(
            {
                "model": meta["model"],
                "dtype": meta["dtype"],
                "gpu": meta["gpu"],
                "device_name": out["device_name"],
                "correct": correct,
                "y_pred": y_pred,
            }
        )
        if logprob is not None:
            b = logit_bundles.setdefault(
                meta["model"], {"device_name": {}, "logprob": {}, "n_classes": n_classes}
            )
            b["device_name"][meta["gpu"]] = out["device_name"]
            b["logprob"][meta["gpu"]] = logprob
        print(f"{meta['model']:8s} {meta['gpu']:6s} {meta['dtype']:5s} acc={out['accuracy']:.4f}")

    if not logits_only:
        _write_json(_out_path("results", dataset), results)
        _write_prediction_log(
            f"predictions-{dataset}.json",
            {
                "dataset": dataset,
                "entrypoint": "main",
                "n_test": len(labels["y_test"]),
                "y_true": labels["y_test"],
                "sensitive": labels["sensitive_test"],
                "predictions": pred_entries,
            },
        )
    for model_name, b in logit_bundles.items():
        gpus = sorted(b["logprob"])
        _write_json(
            os.path.join(RESULTS_DIR, "logits", f"logits-{dataset}-{model_name}-fp32.json"),
            {
                "dataset": dataset,
                "model": model_name,
                "dtype": "fp32",
                "reference_gpu": "T4" if "T4" in gpus else gpus[0],
                "n_test": len(labels["y_test"]),
                "n_classes": b["n_classes"],
                "repeats": logit_repeats,
                "gpus": gpus,
                "device_name": b["device_name"],
                "y_true": labels["y_test"],
                "sensitive": labels["sensitive_test"],
                # per GPU: shape (repeats, n_test, n_classes), log(predict_proba)
                "logprob": {g: b["logprob"][g] for g in gpus},
            },
        )
