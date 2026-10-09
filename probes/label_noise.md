Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=1300, input mean |x|=0.66, input std=0.98
- per-layer weight-gradient norms (first→last): 9.83e-02, 1.48e-01, 1.42e-01
- fraction of first-layer ReLUs inactive on all probe inputs: 0.00

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | 2.122 | 1.95 | 0.372 | 0.602 | 0.049 | 1.93e-01 / 2.49e-01 |
| 2 | 1.862 | 1.157 | 0.455 | 0.807 | 0.105 | 3.14e-01 / 3.88e-01 |
| 3 | 1.734 | 1.013 | 0.494 | 0.841 | 0.0847 | 4.62e-01 / 4.86e-01 |
| 5 | 1.59 | 0.9058 | 0.526 | 0.895 | 0.157 | 5.65e-01 / 4.45e-01 |
| 10 | 1.295 | 0.9248 | 0.584 | 0.817 | 0.209 | 9.65e-01 / 5.08e-01 |
| 15 | 0.9969 | 0.9928 | 0.675 | 0.742 | 0.168 | 1.35e+00 / 5.79e-01 |
| 20 | 0.6693 | 1.296 | 0.787 | 0.616 | 0.104 | 1.39e+00 / 5.29e-01 |
| 30 | 0.2362 | 1.866 | 0.933 | 0.577 | 0.0792 | 1.42e+00 / 5.59e-01 |
