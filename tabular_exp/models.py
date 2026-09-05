"""Model wrappers for the hardware/precision fairness sweep.

Design intent: isolate *inference-time* hardware/precision effects from
training randomness. MLP and XGBoost are trained exactly once on CPU with a
fixed seed and their frozen weights are reused for every hardware/dtype
config. TabPFN needs no training (in-context, pretrained prior) so its
"fit" call is just storing the context; the entire hardware-sensitive
computation is in predict.

XGBoost is included as a near-invariant control: tree traversal is
deterministic and should show ~zero cross-hardware/precision variance,
isolating whether disparities are specific to gradient/attention-based
inference (the mechanism the source paper attributes to gradient flow and
loss-surface geometry).
"""

import inspect
import io
import os

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42, epochs: int = 200) -> bytes:
    """Trains once on CPU, fp32. Returns serialized state_dict bytes so the
    exact same frozen weights can be shipped to every Modal hardware config."""
    torch.manual_seed(seed)
    model = MLP(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def predict_mlp(state_dict_bytes: bytes, n_features: int, X_test: np.ndarray, device: str, dtype: torch.dtype) -> np.ndarray:
    model = MLP(n_features)
    model.load_state_dict(torch.load(io.BytesIO(state_dict_bytes), map_location="cpu"))
    model = model.to(device=device, dtype=dtype)
    model.eval()

    xt = torch.tensor(X_test, dtype=dtype, device=device)
    with torch.no_grad():
        logits = model(xt)
    return logits.float().argmax(dim=1).cpu().numpy()


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> bytearray:
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, random_state=seed, tree_method="hist", n_jobs=1
    )
    model.fit(X_train, y_train)
    return model.get_booster().save_raw(raw_format="ubj")


def predict_xgboost(model_bytes: bytearray, X_test: np.ndarray) -> np.ndarray:
    model = xgb.XGBClassifier()
    model.load_model(bytearray(model_bytes))
    return model.predict(X_test)


# In-context attention is O((context + eval_batch)^2); with the full 20% test
# split (tens of thousands of rows) and a 10k context, a single predict call
# OOMs a 16 GB T4. Chunk the eval rows -- prediction is per-row independent
# given the fixed context, so this is numerically equivalent to one call, and
# the batch size is held constant across GPUs so it stays a controlled factor.
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "1024"))


def _batched(fn, X: np.ndarray, concat_axis: int = 0) -> np.ndarray:
    if len(X) <= EVAL_BATCH:
        return np.asarray(fn(X))
    parts = [np.asarray(fn(X[i : i + EVAL_BATCH])) for i in range(0, len(X), EVAL_BATCH)]
    return np.concatenate(parts, axis=concat_axis)


def _fit_tabpfn(X_train: np.ndarray, y_train: np.ndarray, device: str):
    """TabPFN has no dataset-specific training weights to freeze: `fit` just
    stores the context for in-context prediction. All hardware/precision
    sensitivity lives in the predict/predict_proba forward pass."""
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device=device)
    clf.fit(X_train, y_train)
    return clf


def predict_tabpfn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, device: str, dtype: torch.dtype) -> np.ndarray:
    clf = _fit_tabpfn(X_train, y_train, device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_enabled = dtype != torch.float32
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
        preds = _batched(clf.predict, X_test)
    return np.asarray(preds)


def predict_proba_tabpfn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, device: str) -> np.ndarray:
    """Class probabilities, fp32 only — used for margin/sensitivity analysis
    rather than the hardware/precision sweep."""
    clf = _fit_tabpfn(X_train, y_train, device)
    return np.asarray(clf.predict_proba(X_test))


def _supported_kwargs(fn, **kwargs) -> dict:
    """Subset of ``kwargs`` that ``fn`` actually accepts. TabDPT's
    ``predict`` takes ``n_ensembles``/``seed`` but its ``predict_proba`` (in
    some released versions) does not, so we filter rather than assume."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _autocast(device: str, dtype: torch.dtype):
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype != torch.float32)


def predict_proba_incontext(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    """Class probabilities from an in-context model (tabpfn / tabicl / tabfm /
    tabdpt) under the sweep's precision handling — the pre-softmax input the
    correction-vector scheme operates on. Mirrors the dtype control of the
    matching ``predict_*`` wrapper above (autocast for tabpfn/tabicl/tabdpt,
    load-time cast for tabfm)."""
    if model_name == "tabpfn":
        clf = _fit_tabpfn(X_train, y_train, device)
        with _autocast(device, dtype):
            return _batched(clf.predict_proba, X_eval)

    if model_name == "tabicl":
        from tabicl import TabICLClassifier

        clf = TabICLClassifier(device=device, use_amp=False)
        clf.fit(X_train, y_train)
        with _autocast(device, dtype):
            return _batched(clf.predict_proba, X_eval)

    if model_name == "tabdpt":
        from tabdpt import TabDPTClassifier

        clf = TabDPTClassifier(device=device, use_flash=False, compile=False, verbose=False)
        clf.fit(X_train, y_train)
        kwargs = _supported_kwargs(clf.predict_proba, n_ensembles=1, seed=42)
        with _autocast(device, dtype):
            return _batched(lambda Xb: clf.predict_proba(Xb, **kwargs), X_eval)

    if model_name == "tabfm":
        from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch

        load_dtype = None if dtype == torch.float32 else dtype
        model = tabfm_v1_0_0_pytorch.load(
            model_type="classification", device=device, dtype=load_dtype
        )
        clf = TabFMClassifier(model=model)
        clf.fit(X_train, y_train)
        return _batched(clf.predict_proba, X_eval)

    raise ValueError(f"unknown in-context model {model_name!r}")


def predict_tabfm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, device: str, dtype: torch.dtype) -> np.ndarray:
    """Google Research's TabFM: another zero-shot in-context tabular
    foundation model (like TabPFN/TabICL). `fit` just stores the context;
    the hardware/precision-sensitive work is the single forward pass.

    Unlike TabPFN/TabICL, the PyTorch loader casts the whole model to a
    compute dtype itself (`dtype=None` keeps the fp32 weights, else it's
    designed for bf16), so we pass the sweep dtype there rather than wrapping
    predict in an external `torch.autocast` context."""
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch

    load_dtype = None if dtype == torch.float32 else dtype
    model = tabfm_v1_0_0_pytorch.load(
        model_type="classification", device=device, dtype=load_dtype
    )
    clf = TabFMClassifier(model=model)
    clf.fit(X_train, y_train)
    return _batched(clf.predict, X_test)


def predict_tabdpt(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, device: str, dtype: torch.dtype) -> np.ndarray:
    """Layer6's TabDPT: a retrieval-augmented in-context tabular transformer.
    `fit` stores the context (and builds the kNN index); the
    hardware/precision-sensitive work is the forward pass.

    TabDPT has no compute-dtype knob, so we control precision the same way as
    TabICL — an explicit `torch.autocast` context. `use_flash=False` forces
    SDPA to pick its kernel from the active dtype (the flash kernel is
    bf16/fp16-only and would break the fp32 leg), and `compile=False` removes
    torch.compile from the loop so the only thing varying across configs is
    the hardware and the autocast dtype. `n_ensembles=1` keeps it a single
    deterministic forward pass rather than TabDPT's default 8-way
    class-permutation ensemble."""
    from tabdpt import TabDPTClassifier

    clf = TabDPTClassifier(device=device, use_flash=False, compile=False, verbose=False)
    clf.fit(X_train, y_train)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_enabled = dtype != torch.float32
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
        preds = _batched(lambda Xb: clf.predict(Xb, n_ensembles=1, seed=42), X_test)
    return np.asarray(preds)


def predict_tabicl(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, device: str, dtype: torch.dtype) -> np.ndarray:
    """TabICL is another in-context tabular transformer (like TabPFN): `fit`
    just stores the context, the hardware/precision-sensitive work is in the
    forward pass. `use_amp=False` disables its internal autocast so the
    dtype is controlled by our explicit autocast context, matching TabPFN."""
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device=device, use_amp=False)
    clf.fit(X_train, y_train)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_enabled = dtype != torch.float32
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
        preds = _batched(clf.predict, X_test)
    return np.asarray(preds)
