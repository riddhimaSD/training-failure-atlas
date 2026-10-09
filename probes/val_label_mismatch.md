Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=1300, input mean |x|=0.66, input std=0.98
- per-layer weight-gradient norms (first→last): 1.56e-01, 2.33e-01, 2.22e-01
- fraction of first-layer ReLUs inactive on all probe inputs: 0.00

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | 1.172 | 3.001 | 0.717 | 0.008 | 0.317 | 3.09e-01 / 3.81e-01 |
| 2 | 0.2515 | 9.18 | 0.919 | 0.082 | 0.272 | 4.61e-01 / 5.33e-01 |
| 3 | 0.09411 | 11.35 | 0.975 | 0.107 | 0.0783 | 4.07e-01 / 3.73e-01 |
| 5 | 0.02985 | 12.54 | 0.994 | 0.103 | 0.0281 | 2.67e-01 / 1.91e-01 |
| 10 | 0.005761 | 14.35 | 1.000 | 0.103 | 0.00265 | 5.57e-02 / 3.83e-02 |
| 15 | 0.002987 | 15.41 | 1.000 | 0.103 | 0.00118 | 2.77e-02 / 1.84e-02 |
| 20 | 0.001991 | 16.12 | 1.000 | 0.103 | 0.0008 | 1.92e-02 / 1.23e-02 |
| 30 | 0.001168 | 17.07 | 1.000 | 0.103 | 0.00045 | 1.15e-02 / 7.05e-03 |
