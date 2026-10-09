Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=1300, input mean |x|=0.66, input std=0.98
- per-layer weight-gradient norms (first→last): 0.00e+00, 0.00e+00, 0.00e+00
- fraction of first-layer ReLUs inactive on all probe inputs: 0.88

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | 2.302 | 2.304 | 0.102 | 0.101 | 0.00571 | 0.00e+00 / 0.00e+00 |
| 2 | 2.302 | 2.312 | 0.108 | 0.078 | 0.0079 | 0.00e+00 / 0.00e+00 |
| 3 | 2.301 | 2.309 | 0.108 | 0.078 | 0.00721 | 0.00e+00 / 0.00e+00 |
| 5 | 2.302 | 2.31 | 0.108 | 0.078 | 0.00853 | 0.00e+00 / 0.00e+00 |
| 10 | 2.302 | 2.309 | 0.108 | 0.078 | 0.00692 | 0.00e+00 / 0.00e+00 |
| 15 | 2.302 | 2.311 | 0.102 | 0.089 | 0.00613 | 0.00e+00 / 0.00e+00 |
| 20 | 2.302 | 2.309 | 0.105 | 0.093 | 0.0109 | 0.00e+00 / 0.00e+00 |
| 30 | 2.302 | 2.309 | 0.108 | 0.078 | 0.00785 | 0.00e+00 / 0.00e+00 |
