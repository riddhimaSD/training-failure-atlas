"""Train a small network on sklearn's digits dataset under a named scenario and record training telemetry.

Telemetry per epoch: train/val loss and accuracy, per-layer gradient norms, the spread of mini-batch losses inside the
epoch, and the fraction of dead ReLU units. Also recorded once before any update ("step 0"): per-layer gradient
norms and dead-unit fraction. That's what separates bad initialisation from a bad learning rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn

torch.set_num_threads(1)


@dataclass(frozen=True)
class Config:
    hidden: tuple = (128, 128)
    activation: str = "relu"
    lr: float = 0.05
    epochs: int = 30
    batch_size: int = 64
    shuffle: bool = True
    standardize: bool = True
    input_scale: float = 1.0          # multiply raw pixels (0..16) by this when not standardised
    train_size: int | None = None     # subsample the training set
    label_noise: float = 0.0          # fraction of training labels replaced at random
    init_gain: float = 1.0            # multiplier on default init of hidden weights
    hidden_bias: float = 0.0          # constant init for hidden biases
    val_shift: float = 0.0            # constant added to validation inputs (in standardised units)
    val_from_train: bool = False      # validation rows drawn from the training rows
    val_label_perm: bool = False      # validation labels permuted (wrong label mapping)


@dataclass
class Telemetry:
    epochs: list = field(default_factory=list)
    step0: dict = field(default_factory=dict)
    diverged: bool = False


def _data(cfg: Config, seed: int):
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx].astype(np.float32), y[idx]
    n_tr = 1300
    Xtr, ytr, Xva, yva = X[:n_tr], y[:n_tr].copy(), X[n_tr:], y[n_tr:].copy()
    if cfg.train_size:
        Xtr, ytr = Xtr[:cfg.train_size], ytr[:cfg.train_size]
    if cfg.standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    else:
        Xtr, Xva = Xtr * cfg.input_scale, Xva * cfg.input_scale
    if cfg.label_noise:
        m = rng.random(len(ytr)) < cfg.label_noise
        ytr[m] = rng.integers(0, 10, m.sum())
    if cfg.val_from_train:
        j = rng.choice(len(Xtr), len(Xva), replace=False)
        Xva, yva = Xtr[j].copy(), ytr[j].copy()
    if cfg.val_shift:
        Xva = Xva + cfg.val_shift
    if cfg.val_label_perm:
        yva = rng.permutation(10)[yva]
    if not cfg.shuffle:                    # sorted by class: the classic "forgot to shuffle" loader
        o = np.argsort(ytr, kind="stable")
        Xtr, ytr = Xtr[o], ytr[o]
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)  # noqa: E731
    return t(Xtr), t(ytr, torch.long), t(Xva), t(yva, torch.long)


def _model(cfg: Config, seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    act = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}[cfg.activation]
    layers, d = [], 64
    for h in cfg.hidden:
        lin = nn.Linear(d, h)
        with torch.no_grad():
            lin.weight.mul_(cfg.init_gain)
            lin.bias.fill_(cfg.hidden_bias)
        layers += [lin, act()]
        d = h
    layers.append(nn.Linear(d, 10))
    return nn.Sequential(*layers)


def _linears(m):
    return [l for l in m if isinstance(l, nn.Linear)]


def _grad_norms(m):
    return [float(l.weight.grad.norm()) if l.weight.grad is not None else 0.0 for l in _linears(m)]


def _dead_fraction(m, X):
    """Share of first-hidden-layer ReLU units that output 0 for every probe example."""
    if not isinstance(m[1], nn.ReLU):
        return 0.0
    with torch.no_grad():
        h = m[1](m[0](X))
    return float((h.max(0).values <= 0).float().mean())


def run(cfg: Config, seed: int) -> Telemetry:
    Xtr, ytr, Xva, yva = _data(cfg, seed)
    m = _model(cfg, seed)
    lossf = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(m.parameters(), lr=cfg.lr, momentum=0.9)
    tel = Telemetry()
    m.zero_grad()
    lossf(m(Xtr[:256]), ytr[:256]).backward()
    tel.step0 = {"grad_norms": _grad_norms(m), "dead_fraction": _dead_fraction(m, Xtr[:512]),
                 "input_abs_mean": float(Xtr.abs().mean()), "input_std": float(Xtr.std()), "n_train": len(Xtr)}
    m.zero_grad()
    g = torch.Generator().manual_seed(seed)
    for ep in range(cfg.epochs):
        m.train()
        order = torch.randperm(len(Xtr), generator=g) if cfg.shuffle else torch.arange(len(Xtr))
        batch_losses, gn = [], []
        for i in range(0, len(Xtr), cfg.batch_size):
            j = order[i:i + cfg.batch_size]
            loss = lossf(m(Xtr[j]), ytr[j])
            opt.zero_grad()
            loss.backward()
            gn.append(_grad_norms(m))
            opt.step()
            batch_losses.append(float(loss.detach()))
        m.eval()
        with torch.no_grad():
            out_tr, out_va = m(Xtr), m(Xva)
            rec = {
                "epoch": ep + 1,
                "train_loss": float(lossf(out_tr, ytr)), "val_loss": float(lossf(out_va, yva)),
                "train_acc": float((out_tr.argmax(1) == ytr).float().mean()),
                "val_acc": float((out_va.argmax(1) == yva).float().mean()),
                "batch_loss_std": float(np.std(batch_losses)),
                "grad_norms": np.mean(gn, axis=0).tolist(),
                "dead_fraction": _dead_fraction(m, Xtr[:512]),
            }
        tel.epochs.append(rec)
        if not all(math.isfinite(rec[k]) for k in ("train_loss", "val_loss")) or rec["train_loss"] > 1e4:
            tel.diverged = True
            break
    return tel


HEALTHY = Config()
