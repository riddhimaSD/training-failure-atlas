Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=1300, input mean |x|=0.66, input std=0.98
- per-layer weight-gradient norms (first→last): 2.46e-07, 6.92e-07, 5.62e-06, 3.55e-05, 2.31e-04, 1.75e-03, 1.36e-02, 7.86e-02, 5.50e-01
- fraction of first-layer ReLUs inactive on all probe inputs: 0.00

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | 2.308 | 2.306 | 0.097 | 0.111 | 0.0278 | 2.75e-07 / 5.78e-01 |
| 2 | 2.327 | 2.312 | 0.092 | 0.109 | 0.0192 | 2.67e-07 / 5.47e-01 |
| 3 | 2.306 | 2.313 | 0.105 | 0.091 | 0.0385 | 2.72e-07 / 5.58e-01 |
| 5 | 2.334 | 2.332 | 0.097 | 0.111 | 0.0351 | 2.67e-07 / 5.23e-01 |
| 10 | 2.309 | 2.319 | 0.102 | 0.101 | 0.0226 | 2.48e-07 / 4.02e-01 |
| 15 | 2.311 | 2.32 | 0.105 | 0.091 | 0.0188 | 2.15e-07 / 2.86e-01 |
| 20 | 2.304 | 2.303 | 0.098 | 0.105 | 0.0179 | 2.00e-07 / 2.70e-01 |
| 30 | 2.302 | 2.307 | 0.105 | 0.093 | 0.0164 | 1.75e-07 / 1.90e-01 |
