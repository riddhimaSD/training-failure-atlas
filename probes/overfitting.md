Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=100, input mean |x|=0.65, input std=0.92
- per-layer weight-gradient norms (first→last): 2.00e-01, 5.78e-01, 5.50e-01
- fraction of first-layer ReLUs inactive on all probe inputs: 0.00

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | 2.186 | 255.9 | 0.430 | 0.284 | 0.00583 | 2.39e-01 / 6.46e-01 |
| 2 | 1.995 | 554.5 | 0.450 | 0.268 | 0.0436 | 2.32e-01 / 5.72e-01 |
| 3 | 1.746 | 1138 | 0.440 | 0.258 | 0.0612 | 2.45e-01 / 6.04e-01 |
| 5 | 1.215 | 2244 | 0.660 | 0.390 | 0.0422 | 2.83e-01 / 6.46e-01 |
| 10 | 0.1602 | 8187 | 1.000 | 0.877 | 0.0361 | 2.01e-01 / 3.32e-01 |
| 15 | 0.01718 | 1.259e+04 | 1.000 | 0.859 | 0.00613 | 4.69e-02 / 6.04e-02 |
| 20 | 0.003562 | 1.325e+04 | 1.000 | 0.865 | 0.000307 | 1.70e-02 / 1.81e-02 |
| 30 | 0.001191 | 1.335e+04 | 1.000 | 0.869 | 0.000155 | 3.95e-03 / 4.00e-03 |
| 40 | 0.0008927 | 1.309e+04 | 1.000 | 0.871 | 0.000186 | 3.24e-03 / 3.42e-03 |
| 60 | 0.0006955 | 1.283e+04 | 1.000 | 0.867 | 0.000264 | 2.53e-03 / 2.44e-03 |
