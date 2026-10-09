"""Run every scenario on dev and held-out seeds, score the diagnostician, and draw the atlas.

    python evaluate.py                      # ~2-3 min on a laptop CPU (parallel)
Writes results/: telemetry.jsonl, diagnosis.json, summary.md, atlas.png
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from atlas.core import run
from atlas.diagnose import diagnose, diagnose_v1
from atlas.scenarios import SCENARIOS

DEV, HELD_OUT, FRESH = [0, 1, 2], list(range(100, 110)), list(range(200, 210))


def _one(args):
    name, seed = args
    t = run(SCENARIOS[name][0], seed)
    v1, why1 = diagnose_v1(t)
    v2, why2 = diagnose(t)
    return {"scenario": name, "seed": seed, "predicted_v1": v1, "evidence_v1": why1,
            "predicted": v2, "evidence": why2, "telemetry": asdict(t)}


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path("results")
    out.mkdir(exist_ok=True)
    jobs = [(n, s) for n in SCENARIOS for s in DEV + HELD_OUT + FRESH]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_one, jobs))
    with (out / "telemetry.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def score(seeds, key):
        sub = [r for r in rows if r["seed"] in seeds]
        return sum(r[key] == r["scenario"] for r in sub), len(sub), sub

    d1, dn, _ = score(DEV, "predicted_v1")
    h1, hn, held = score(HELD_OUT, "predicted_v1")
    f2, fn, fresh = score(FRESH, "predicted")
    per1 = {n: sum(r["predicted_v1"] == n for r in held if r["scenario"] == n) for n in SCENARIOS}
    per2 = {n: sum(r["predicted"] == n for r in fresh if r["scenario"] == n) for n in SCENARIOS}
    conf1 = Counter((r["scenario"], r["predicted_v1"]) for r in held if r["predicted_v1"] != r["scenario"])
    conf2 = Counter((r["scenario"], r["predicted"]) for r in fresh if r["predicted"] != r["scenario"])
    (out / "diagnosis.json").write_text(json.dumps({
        "v1_dev": {"correct": d1, "total": dn}, "v1_held_out_100_109": {"correct": h1, "total": hn},
        "v2_fresh_200_209": {"correct": f2, "total": fn}, "per_scenario_v1_held_out": per1, "per_scenario_v2_fresh": per2,
        "confusions_v1": [{"true": t, "predicted": p, "count": c} for (t, p), c in conf1.items()],
        "confusions_v2": [{"true": t, "predicted": p, "count": c} for (t, p), c in conf2.items()],
    }, indent=1))

    lines = [f"**v1 rules** (written on dev seeds 0-2, then frozen): dev {d1}/{dn}, **held-out seeds 100-109: "
             f"{h1}/{hn} ({h1 / hn:.0%})**.  ",
             f"**v2 rules** (v1 + one rule added after analysing v1's errors): **fresh seeds 200-209: {f2}/{fn} ({f2 / fn:.0%})**.", "",
             "| Scenario | v1 held-out | v2 fresh | Evidence the diagnostician cites (v2, first fresh seed) |",
             "|---|---|---|---|"]
    for n in SCENARIOS:
        ev = next(r["evidence"] for r in fresh if r["scenario"] == n)
        lines.append(f"| `{n}` | {per1[n]}/{len(HELD_OUT)} | {per2[n]}/{len(FRESH)} | {ev} |")
    if conf1:
        lines += ["", "v1 confusions (held-out): " + "; ".join(f"`{t}` → `{p}` ×{c}" for (t, p), c in conf1.items())]
    lines += ["", "v2 confusions (fresh): " + ("; ".join(f"`{t}` → `{p}` ×{c}" for (t, p), c in conf2.items()) or "none")]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atlas(rows, out)
    print("\n".join(lines))


def _atlas(rows, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(SCENARIOS)
    fig, axes = plt.subplots(3, 5, figsize=(17, 9.5))
    for ax, n in zip(axes.flat, names):
        for r in [r for r in rows if r["scenario"] == n and r["seed"] in DEV]:
            e = r["telemetry"]["epochs"]
            x = [p["epoch"] for p in e]
            mk = "o" if len(e) < 3 else None
            ax.plot(x, [p["train_acc"] for p in e], color="#1f4e9a", lw=1, alpha=0.8, marker=mk)
            ax.plot(x, [p["val_acc"] for p in e], color="#d08c2c", lw=1, alpha=0.8, marker=mk)
            if r["telemetry"]["diverged"] and len(e) < 3:
                ax.text(0.5, 0.5, "diverged in epoch 1", transform=ax.transAxes, ha="center", fontsize=9, color="#a33")
                ax.set_xlim(0, 30)
        ax.set_title(n.replace("_", " "), fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[len(names)].plot([], [], color="#1f4e9a", label="train accuracy")
    axes.flat[len(names)].plot([], [], color="#d08c2c", label="validation accuracy")
    axes.flat[len(names)].legend(loc="center", fontsize=11)
    fig.suptitle("Training-failure atlas: accuracy curves per scenario (3 dev seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "atlas.png", dpi=110)


if __name__ == "__main__":
    import sys as _s
    if "--plot-only" in _s.argv:
        _atlas([json.loads(l) for l in open("results/telemetry.jsonl", encoding="utf-8")], Path("results"))
    else:
        main()
