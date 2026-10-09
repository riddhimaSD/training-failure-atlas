"""Turn telemetry into "diagnose this run" prompts for a language model, and grade the answers.

    python probes.py --write                 # writes probes/*.md (one per scenario, held-out seed 100)
    python probes.py --run claude-opus-5-5   # needs ANTHROPIC_API_KEY; saves raw answers + scores

The prompt shows only what a practitioner would see in their logs: step-0 stats and a per-epoch table.
The scenario name never appears. Grading checks for the scenario's key concepts (any-of phrases); the raw
answers are saved for human review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUBRIC = {
    "healthy": [["healthy", "no problem", "no issue", "nothing wrong", "normal", "converg"]],
    "lr_too_high": [["learning rate", "lr", "step size"], ["too high", "too large", "reduce", "lower"]],
    "lr_too_low": [["learning rate", "lr", "step size"], ["too low", "too small", "increase", "raise"]],
    "overfitting": [["overfit"]],
    "label_noise": [["label noise", "noisy label", "mislabel", "wrong label", "corrupt"]],
    "unshuffled": [["shuffl", "sorted", "ordered", "class order"]],
    "unnormalized_inputs": [["normali", "standardi", "scale"], ["input"]],
    "dead_relus": [["dead", "dying"], ["relu"]],
    "vanishing_gradients": [["vanishing"]],
    "exploding_init": [["exploding", "explod"], ["init"]],
    "val_distribution_shift": [["shift", "distribution", "preprocessing mismatch", "skew", "mismatch"]],
    "val_leakage": [["leak", "overlap", "contaminat", "duplicate"]],
    "val_label_mismatch": [["label"], ["mapping", "permut", "encod", "mismatch", "misalign", "swapped"]],
}

Q = ("Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). "
     "What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be "
     "confused with, a quick check that would confirm it, and the fix.")


def render(row: dict) -> str:
    t = row["telemetry"]
    s0 = t["step0"]
    lines = [Q, "", "Before training (step 0):",
             f"- n_train={s0['n_train']}, input mean |x|={s0['input_abs_mean']:.2f}, input std={s0['input_std']:.2f}",
             f"- per-layer weight-gradient norms (first→last): {', '.join(f'{g:.2e}' for g in s0['grad_norms'])}",
             f"- fraction of first-layer ReLUs inactive on all probe inputs: {s0['dead_fraction']:.2f}", "",
             "| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |",
             "|---|---|---|---|---|---|---|"]
    e = t["epochs"]
    keep = sorted(set([0, 1, 2, 4, 9, 14, 19, 29, 39, 59, len(e) - 1]) & set(range(len(e))))
    for i in keep:
        r = e[i]
        lines.append(f"| {r['epoch']} | {r['train_loss']:.4g} | {r['val_loss']:.4g} | {r['train_acc']:.3f} | "
                     f"{r['val_acc']:.3f} | {r['batch_loss_std']:.3g} | {r['grad_norms'][0]:.2e} / {r['grad_norms'][-1]:.2e} |")
    if t["diverged"]:
        lines.append(f"\nTraining stopped after epoch {e[-1]['epoch']}: loss became non-finite or exceeded 1e4.")
    return "\n".join(lines)


def grade(scenario: str, text: str) -> bool:
    t = text.lower()
    return all(any(p in t for p in group) for group in RUBRIC[scenario])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open("results/telemetry.jsonl", encoding="utf-8")]
    picks = {r["scenario"]: r for r in rows if r["seed"] == 100}
    out = Path("probes")
    out.mkdir(exist_ok=True)
    if a.write:
        for name, r in picks.items():
            (out / f"{name}.md").write_text(render(r) + "\n", encoding="utf-8")
        print(f"wrote {len(picks)} probes")
    if a.run:
        import anthropic
        client = anthropic.Anthropic()
        res = []
        for name, r in picks.items():
            msg = client.messages.create(model=a.run, max_tokens=4000, messages=[{"role": "user", "content": render(r)}])
            text = "".join(b.text for b in msg.content if b.type == "text")
            res.append({"scenario": name, "pass": grade(name, text), "answer": text})
            print(f"{name:24s} {'PASS' if res[-1]['pass'] else 'FAIL'}")
        (out / f"answers_{a.run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"{sum(x['pass'] for x in res)}/{len(res)} diagnosed (keyword rubric; review answers by hand)")


if __name__ == "__main__":
    main()
