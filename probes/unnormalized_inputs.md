Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=1300, input mean |x|=78.18, input std=96.28
- per-layer weight-gradient norms (first→last): 9.45e+01, 1.62e+02, 1.97e+02
- fraction of first-layer ReLUs inactive on all probe inputs: 0.02

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | 82.5 | 94.77 | 0.105 | 0.093 | 3.68e+04 | 3.65e+03 / 1.17e+03 |
| 2 | 2.471 | 2.545 | 0.103 | 0.089 | 21.2 | 2.04e-03 / 7.30e-01 |
| 3 | 2.332 | 2.368 | 0.105 | 0.091 | 0.0995 | 1.29e+00 / 1.63e-03 |
| 5 | 2.302 | 2.309 | 0.108 | 0.078 | 0.0124 | 0.00e+00 / 0.00e+00 |
| 10 | 2.302 | 2.309 | 0.108 | 0.078 | 0.00693 | 0.00e+00 / 0.00e+00 |
| 15 | 2.302 | 2.311 | 0.102 | 0.089 | 0.00613 | 0.00e+00 / 0.00e+00 |
| 20 | 2.302 | 2.309 | 0.105 | 0.093 | 0.0109 | 0.00e+00 / 0.00e+00 |
| 30 | 2.302 | 2.309 | 0.108 | 0.078 | 0.00785 | 0.00e+00 / 0.00e+00 |
