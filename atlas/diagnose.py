"""A rule-based diagnostician: telemetry in, failure label out.

Rules were written by inspecting development seeds 0-2 only, then frozen. `evaluate.py` scores them on held-out
seeds 100-109. The order matters: checks that need no training (step-0 statistics) come first, because they're
the cheapest and the least ambiguous signals a practitioner has.
"""
from __future__ import annotations

from .core import Telemetry


def diagnose_v1(t: Telemetry) -> tuple[str, str]:
    """Frozen v1 rules (scored on held-out seeds 100-109)."""
    return _diagnose(t, v2=False)


def diagnose(t: Telemetry) -> tuple[str, str]:
    """v2 = v1 + one rule for a too-high learning rate that stalls instead of diverging (scored on fresh seeds 200-209)."""
    return _diagnose(t, v2=True)


def _diagnose(t: Telemetry, v2: bool) -> tuple[str, str]:
    s0, e = t.step0, t.epochs
    g_first, g_last = s0["grad_norms"][0], s0["grad_norms"][-1]
    last = e[-1]

    # --- before any update
    if s0["input_std"] > 5:
        return "unnormalized_inputs", f"input std {s0['input_std']:.1f} (expected ~1)"
    if s0["dead_fraction"] > 0.5:
        return "dead_relus", f"{s0['dead_fraction']:.0%} of first-layer ReLUs inactive on every input at init"
    if max(s0["grad_norms"]) > 20:
        return "exploding_init", f"step-0 gradient norm {max(s0['grad_norms']):.0f} before any update"
    if g_last > 0 and g_first / g_last < 1e-4:
        return "vanishing_gradients", f"first-layer / last-layer gradient ratio {g_first / g_last:.1e} at init"

    # --- during training
    if t.diverged:
        return "lr_too_high", "loss became non-finite or exploded, although step-0 gradients were normal"
    if last["train_acc"] > 0.9 and last["val_acc"] < 0.2:
        return "val_label_mismatch", f"train acc {last['train_acc']:.2f} but val acc {last['val_acc']:.2f} (≈ chance)"
    if last["train_acc"] > 0.99 and abs(last["train_acc"] - last["val_acc"]) < 0.005:
        return "val_leakage", "validation accuracy tracks training accuracy exactly at 100%"
    spread = max((r["batch_loss_std"] for r in e[3:]), default=0.0)
    if spread > 0.5:
        return "unshuffled", f"mini-batch loss std within an epoch peaks at {spread:.2f} after epoch 3 (healthy < 0.1)"
    if any(r["val_acc"] > r["train_acc"] + 0.15 for r in e[:10]):
        return "label_noise", "validation accuracy exceeds training accuracy early in training"
    if v2:
        g0 = s0["grad_norms"][0]
        g_ep1, g_end = e[0]["grad_norms"][0], last["grad_norms"][0]
        if g0 > 0 and (g_ep1 > 10 * g0 or g_end < 1e-6 * g0):
            return "lr_too_high", (f"no divergence, but epoch-1 gradients were {g_ep1 / g0:.0f}x the step-0 gradients and "
                                   f"then collapsed to {g_end:.1e}: the large steps killed the network")
    if (e[0]["train_loss"] - last["train_loss"]) / e[0]["train_loss"] < 0.25:
        return "lr_too_low", "training loss fell by <25% over the whole run, gradients healthy"
    if last["train_acc"] > 0.99 and last["train_acc"] - last["val_acc"] > 0.05:
        early = e[min(2, len(e) - 1)]
        if early["train_acc"] > 0.9 and early["train_acc"] - early["val_acc"] > 0.2:
            return "val_distribution_shift", "train/val gap is large from the first epochs, not growing over time"
        return "overfitting", "train acc reaches 100% while the val gap opens as training continues"
    return "healthy", "no failure signature"
