Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=1300, input mean |x|=0.66, input std=0.98
- per-layer weight-gradient norms (first→last): 1.56e-01, 2.33e-01, 2.22e-01
- fraction of first-layer ReLUs inactive on all probe inputs: 0.00

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | 2.304 | 2.298 | 0.160 | 0.163 | 0.0195 | 2.14e-01 / 3.06e-01 |
| 2 | 2.3 | 2.294 | 0.176 | 0.177 | 0.0151 | 2.08e-01 / 2.93e-01 |
| 3 | 2.295 | 2.29 | 0.188 | 0.189 | 0.0122 | 2.10e-01 / 2.94e-01 |
| 5 | 2.285 | 2.281 | 0.213 | 0.209 | 0.0156 | 2.13e-01 / 3.07e-01 |
| 10 | 2.261 | 2.258 | 0.293 | 0.288 | 0.0153 | 2.16e-01 / 3.06e-01 |
| 15 | 2.236 | 2.235 | 0.371 | 0.376 | 0.0121 | 2.17e-01 / 2.94e-01 |
| 20 | 2.21 | 2.211 | 0.432 | 0.441 | 0.0224 | 2.24e-01 / 3.08e-01 |
| 30 | 2.152 | 2.158 | 0.518 | 0.517 | 0.0201 | 2.45e-01 / 3.19e-01 |
