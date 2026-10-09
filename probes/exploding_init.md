Below is telemetry from a training run (a small MLP on a 10-class image dataset, SGD with momentum). What, if anything, is wrong? Give the single most likely diagnosis, the evidence for it, what it could be confused with, a quick check that would confirm it, and the fix.

Before training (step 0):
- n_train=1300, input mean |x|=0.66, input std=0.98
- per-layer weight-gradient norms (first→last): 5.53e+01, 1.20e+02, 1.77e+02, 2.11e+02, 2.49e+02, 3.44e+02, 3.37e+02, 3.43e+02, 1.86e+03
- fraction of first-layer ReLUs inactive on all probe inputs: 0.00

| epoch | train loss | val loss | train acc | val acc | batch-loss std | grad norm (first / last layer) |
|---|---|---|---|---|---|---|
| 1 | nan | nan | 0.098 | 0.103 | nan | nan / nan |

Training stopped after epoch 1: loss became non-finite or exceeded 1e4.
