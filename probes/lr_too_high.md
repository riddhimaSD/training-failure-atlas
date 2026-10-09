Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=1300, input mean |x|=0.66, input std=0.98
- per-layer weight-gradient norms (first→last): 1.56e-01, 2.33e-01, 2.22e-01
- fraction of first-layer ReLUs inactive on all probe inputs: 0.00

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | 2.64 | 2.667 | 0.102 | 0.089 | 9.91 | 2.94e+00 / 3.25e+00 |
| 2 | 2.383 | 2.378 | 0.105 | 0.091 | 42.4 | 9.13e+00 / 4.06e-01 |
| 3 | 2.403 | 2.388 | 0.102 | 0.101 | 0.0732 | 0.00e+00 / 0.00e+00 |
| 5 | 2.368 | 2.372 | 0.092 | 0.109 | 0.0405 | 0.00e+00 / 0.00e+00 |
| 10 | 2.363 | 2.381 | 0.098 | 0.105 | 0.0645 | 0.00e+00 / 0.00e+00 |
| 15 | 2.33 | 2.339 | 0.098 | 0.105 | 0.0275 | 0.00e+00 / 0.00e+00 |
| 20 | 2.354 | 2.337 | 0.097 | 0.111 | 0.0673 | 0.00e+00 / 0.00e+00 |
| 30 | 2.391 | 2.453 | 0.102 | 0.089 | 0.0616 | 0.00e+00 / 0.00e+00 |
