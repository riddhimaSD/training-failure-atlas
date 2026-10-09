**v1 rules** (written on dev seeds 0-2, then frozen): dev 39/39, **held-out seeds 100-109: 123/130 (95%)**.  
**v2 rules** (v1 + one rule added after analysing v1's errors): **fresh seeds 200-209: 124/130 (95%)**.

| Scenario | v1 held-out | v2 fresh | Evidence the diagnostician cites (v2, first fresh seed) |
|---|---|---|---|
| `healthy` | 10/10 | 10/10 | no failure signature |
| `lr_too_high` | 3/10 | 10/10 | loss became non-finite or exploded, although step-0 gradients were normal |
| `lr_too_low` | 10/10 | 10/10 | training loss fell by <25% over the whole run, gradients healthy |
| `overfitting` | 10/10 | 10/10 | train acc reaches 100% while the val gap opens as training continues |
| `label_noise` | 10/10 | 10/10 | validation accuracy exceeds training accuracy early in training |
| `unshuffled` | 10/10 | 10/10 | mini-batch loss std within an epoch peaks at 10.23 after epoch 3 (healthy < 0.1) |
| `unnormalized_inputs` | 10/10 | 10/10 | input std 96.1 (expected ~1) |
| `dead_relus` | 10/10 | 10/10 | 85% of first-layer ReLUs inactive on every input at init |
| `vanishing_gradients` | 10/10 | 10/10 | first-layer / last-layer gradient ratio 5.5e-07 at init |
| `exploding_init` | 10/10 | 10/10 | step-0 gradient norm 1510 before any update |
| `val_distribution_shift` | 10/10 | 7/10 | train/val gap is large from the first epochs, not growing over time |
| `val_leakage` | 10/10 | 10/10 | validation accuracy tracks training accuracy exactly at 100% |
| `val_label_mismatch` | 10/10 | 7/10 | train/val gap is large from the first epochs, not growing over time |

v1 confusions (held-out): `lr_too_high` → `lr_too_low` ×4; `lr_too_high` → `healthy` ×2; `lr_too_high` → `unshuffled` ×1

v2 confusions (fresh): `val_distribution_shift` → `overfitting` ×3; `val_label_mismatch` → `val_distribution_shift` ×3
