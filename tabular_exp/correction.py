"""Softmax-input correction vector for cross-hardware consistency.

Scheme (TabPFN for now): TabPFN's last layer is a softmax over per-class
scores. We add a single learned correction vector ``c`` (length = n_classes),
*shared across every GPU*, to those scores and then re-softmax. ``c`` is fit
by a small two-objective optimisation on the dedicated 20% calibration
split (disjoint from both the in-context training set and the test set, so
the calibration logprobs are not leak-inflated):

  1. keep accuracy as high as possible -- cross-entropy of
     ``softmax(score + c)`` against the labels, averaged over GPUs;
  2. minimise cross-GPU accuracy differences -- the variance across GPUs of
     the *soft* per-GPU accuracy (mean probability mass placed on the true
     class).

Both objectives are normalised by their value at ``c = 0`` so a single knob
``acc_weight`` in [0, 1] traces the accuracy/consistency trade-off.

No TabPFN internals are touched: the pre-softmax score we correct is
``log(predict_proba)``, which is exactly the softmax-invariant part of the
true logits, so ``softmax(log p + c)`` is precisely "shift the logits by c".

Because ``c`` is identical on every GPU it cannot make one GPU more accurate
than another in isolation; what it can do is move the decision boundary out
of the dense near-tie region where floating-point non-determinism flips
predictions, which is what drives the cross-GPU spread.
"""

import numpy as np


def centered_logit_l2_drift(
    logprob_by_gpu: dict[str, np.ndarray], ref: str = "T4"
) -> dict[str, float]:
    """Mean over rows of the L2 distance between each GPU's softmax-input and
    the reference GPU's, after row-centering both (softmax is shift-invariant,
    so only the centered part is meaningful). ``logprob_by_gpu`` maps GPU name
    -> ``(N, C)`` ``log(proba)`` array. The reference GPU's own entry is 0."""
    def centered(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        return a - a.mean(axis=-1, keepdims=True)

    r = centered(logprob_by_gpu[ref])
    return {
        g: float(np.linalg.norm(centered(lp) - r, axis=-1).mean())
        for g, lp in logprob_by_gpu.items()
    }


def fit_correction_vector(
    logprob_by_gpu: dict[str, np.ndarray],
    y: np.ndarray,
    acc_weight: float = 0.5,
    steps: int = 1000,
    lr: float = 0.05,
) -> np.ndarray:
    """Fit the correction vector on calibration pre-softmax scores.

    ``logprob_by_gpu`` maps GPU name -> ``(N, C)`` array of ``log(proba)`` on
    that GPU for the calibration rows; ``y`` is the ``(N,)`` label vector.
    ``acc_weight`` weights objective 1 vs. objective 2 (see module docstring).
    """
    import torch

    gpus = sorted(logprob_by_gpu)
    L = torch.tensor(np.stack([logprob_by_gpu[g] for g in gpus]), dtype=torch.float32)  # (G, N, C)
    yt = torch.tensor(np.asarray(y), dtype=torch.long)
    rows = torch.arange(yt.shape[0])

    c = torch.zeros(L.shape[-1], requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)

    def objectives(cv):
        logp = torch.log_softmax(L + cv, dim=-1)  # (G, N, C)
        true_logp = logp[:, rows, yt]  # (G, N)
        ce = -true_logp.mean()  # accuracy surrogate (lower is better)
        soft_acc = true_logp.exp().mean(dim=1)  # (G,) soft per-GPU accuracy
        spread = soft_acc.var(unbiased=False)  # cross-GPU accuracy diff
        return ce, spread

    with torch.no_grad():
        ce0, spread0 = objectives(c)
    ce0 = ce0.clamp_min(1e-8)
    spread0 = spread0.clamp_min(1e-12)

    for _ in range(steps):
        opt.zero_grad()
        ce, spread = objectives(c)
        loss = acc_weight * (ce / ce0) + (1.0 - acc_weight) * (spread / spread0)
        loss.backward()
        opt.step()

    return c.detach().numpy()


def apply_correction(logprob: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return ``softmax(logprob + c)`` row-wise (numerically stable)."""
    z = np.asarray(logprob, dtype=np.float64) + np.asarray(c, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)
