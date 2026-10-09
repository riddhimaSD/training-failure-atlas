Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=1300, input mean |x|=0.66, input std=0.98
- per-layer weight-gradient norms (first→last): 3.18e-01, 6.14e-01, 8.25e-01
- fraction of first-layer ReLUs inactive on all probe inputs: 0.00

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | 2.142 | 2.139 | 0.418 | 0.429 | 0.267 | 5.29e-01 / 1.08e+00 |
| 2 | 1.583 | 1.564 | 0.478 | 0.477 | 0.151 | 5.20e-01 / 8.90e-01 |
| 3 | 1.571 | 1.522 | 0.538 | 0.555 | 0.569 | 1.69e+00 / 2.10e+00 |
| 5 | 1.362 | 1.246 | 0.665 | 0.700 | 1.9 | 2.50e+00 / 1.87e+00 |
| 10 | 0.9519 | 0.9285 | 0.688 | 0.702 | 0.956 | 1.04e+00 / 8.51e-01 |
| 15 | 0.6677 | 0.7081 | 0.763 | 0.773 | 0.648 | 8.54e-01 / 5.34e-01 |
| 20 | 0.4083 | 0.523 | 0.890 | 0.885 | 0.635 | 5.26e-01 / 1.93e-01 |
| 30 | 0.1858 | 0.3608 | 0.948 | 0.940 | 0.432 | 2.43e-01 / 1.71e-01 |
