"""Fast checks of the documented signatures on dev seed 0 (the full evaluation lives in evaluate.py)."""
import pytest

from atlas.core import run
from atlas.diagnose import diagnose, diagnose_v1
from atlas.scenarios import SCENARIOS
from probes import grade, render


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_dev_seed_diagnosed(name):
    assert diagnose(run(SCENARIOS[name][0], 0))[0] == name


def test_step0_signatures():
    assert run(SCENARIOS["dead_relus"][0], 0).step0["dead_fraction"] > 0.5
    g = run(SCENARIOS["vanishing_gradients"][0], 0).step0["grad_norms"]
    assert g[0] / g[-1] < 1e-4
    assert run(SCENARIOS["unnormalized_inputs"][0], 0).step0["input_std"] > 5


def test_label_noise_val_beats_train_early():
    e = run(SCENARIOS["label_noise"][0], 0).epochs
    assert any(r["val_acc"] > r["train_acc"] + 0.15 for r in e[:10])


def test_stalled_high_lr_is_the_v1_blind_spot():
    t = run(SCENARIOS["lr_too_high"][0], 100)       # this seed stalls instead of diverging
    assert not t.diverged
    assert diagnose_v1(t)[0] == "lr_too_low" and diagnose(t)[0] == "lr_too_high"


def test_probe_render_hides_label_and_grades():
    t = run(SCENARIOS["overfitting"][0], 0)
    from dataclasses import asdict
    text = render({"telemetry": asdict(t)})
    assert "overfitting" not in text.lower() and "| epoch |" in text
    assert grade("overfitting", "Classic overfitting: train 100%, val loss rising.")
    assert not grade("lr_too_low", "The learning rate is too high")
