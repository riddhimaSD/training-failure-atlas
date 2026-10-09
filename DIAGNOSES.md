# Diagnosis guide

For each failure: **the signature in the telemetry**, **what it's commonly mistaken for**, **a cheap test that confirms it**, and **the fix**. Each entry is grounded in the runs in `results/telemetry.jsonl`.

---

### Learning rate too high (`lr_too_high`)
- **Signature:** usually the loss goes non-finite in the first epoch. *But not always.* On 7 of 10 held-out seeds, LR 1.5 did not diverge. The epoch-1 gradient norm jumped 15–300× above its value at initialisation, the loss rose above its starting value (2.6–3.8 vs ln 10 ≈ 2.30), and then the gradients collapsed to exactly 0. The large steps killed the network, and the loss sat flat at ~2.35 for the rest of the run.
- **Commonly mistaken for:** *learning rate too low.* Both show a flat loss near ln(#classes). This is exactly the error rule set v1 made.
- **Confirm:** compare the epoch-1 gradient norm with the step-0 gradient norm, and check whether gradients are exactly 0 later. A too-low LR keeps steady, non-zero gradients (~0.23 here). Or run a short LR range test.
- **Fix:** lower the LR (here 0.05 works), add warmup, and consider gradient clipping.

### Learning rate too low (`lr_too_low`)
- **Signature:** the loss decreases monotonically but very slowly (2.30 → 2.15 in 30 epochs), with healthy, stable gradient norms and no dead units.
- **Commonly mistaken for:** a too-high LR that stalled (see above), "the model is too small", or "the task is too hard".
- **Confirm:** a 10× LR increase on a few hundred steps makes the loss fall much faster.
- **Fix:** raise the LR, or use an LR finder / one-cycle schedule.

### Overfitting (`overfitting`)
- **Signature:** train accuracy reaches 100%. Validation loss reaches its minimum early (epoch 2 with 100 examples) and then rises, while validation accuracy plateaus well below train.
- **Commonly mistaken for:** distribution shift (the final curves can look identical, see below) or label noise.
- **Confirm:** the gap shrinks as training data grows, and early-stopping at the val-loss minimum helps.
- **Fix:** more data, augmentation, weight decay or dropout, a smaller model, early stopping.

### Label noise (`label_noise`)
- **Signature:** **validation accuracy exceeds training accuracy early on** (0.89 val vs 0.50 train at epoch 3 with 50% noise). The clean val labels reward the true function the network learns first. Later the network memorises the noise: train accuracy climbs and val accuracy falls.
- **Commonly mistaken for:** overfitting, or "the validation set is easier".
- **Confirm:** inspect the highest-loss training examples; many have wrong labels. Clean or relabel a sample and retrain.
- **Fix:** relabel or filter (loss-based / confident learning), use robust losses, and stop early before memorisation.

### Data not shuffled (`unshuffled`)
- **Signature:** large spread of mini-batch losses *within* an epoch (std 0.7–10 after epoch 3 vs < 0.1 when healthy). Each batch contains one class, so the model swings between classes. Final accuracy is *sometimes fine*, so end-of-training metrics hide it.
- **Commonly mistaken for:** an LR that's too high, or "noisy data".
- **Confirm:** print the label histogram of a few consecutive batches.
- **Fix:** `shuffle=True` (or a shuffled sampler), and reshuffle every epoch.

### Unnormalised inputs (`unnormalized_inputs`)
- **Signature:** input std ≈ 96 instead of ≈ 1, step-0 gradient norms ~100× normal, and the loss stuck at chance.
- **Commonly mistaken for:** a bad learning rate or bad initialisation.
- **Confirm:** look at the data. `X.mean()`, `X.std()` per feature *as the model receives it*.
- **Fix:** standardise using statistics computed on the training split only, and apply the same transform at inference.

### Dead ReLUs (`dead_relus`)
- **Signature:** 85–90% of first-layer ReLUs output 0 for every input at initialisation, gradients are 0, and the loss is flat at exactly ln 10.
- **Commonly mistaken for:** an LR that's too low (flat loss), or a bug in the data pipeline.
- **Confirm:** measure the dead-unit fraction on a probe batch at step 0.
- **Fix:** sensible bias init (0), He init, LeakyReLU/GELU, a lower LR (high LRs also kill ReLUs mid-training).

### Vanishing gradients (`vanishing_gradients`)
- **Signature:** at init the first-layer gradient is ~10⁻⁷ of the last layer's (8 sigmoid layers). The loss is flat because only the last layer learns.
- **Commonly mistaken for:** "the network is too small" (so people add layers and make it worse).
- **Confirm:** per-layer gradient norms at step 0.
- **Fix:** ReLU-family activations, normalisation layers, residual connections, proper init.

### Exploding gradients from initialisation (`exploding_init`)
- **Signature:** step-0 gradient norm ~1,500–2,300 *before any update*, then immediate divergence.
- **Commonly mistaken for:** an LR that's too high. Both diverge, but here the gradients are huge before the optimizer has done anything.
- **Confirm:** step-0 gradient norms and activation scales per layer.
- **Fix:** variance-preserving init (He/Xavier), normalisation, gradient clipping.

### Validation distribution shift (`val_distribution_shift`)
- **Signature:** a large train/val gap from the first epochs (val accuracy 0.25–0.80 while train is near 100%) that does not come from growing overfitting.
- **Commonly mistaken for:** **overfitting.** On 3 of 10 fresh seeds the gap at epoch 3 was below the threshold and grew over time, making the accuracy curves indistinguishable from overfitting. Accuracy curves alone can't separate these two.
- **Confirm:** compare input statistics of train vs val (mean and std per feature, or a domain classifier). Here the val inputs are shifted by +3 std. *(v3 of the diagnostician will record val input statistics.)*
- **Fix:** the same preprocessing for train and val/serving, the same pipeline object, and monitoring for train/serve skew.

### Validation leakage (`val_leakage`)
- **Signature:** validation accuracy tracks training accuracy *exactly*, reaching 100%. That's too good for a real held-out set.
- **Commonly mistaken for:** "the model is just really good".
- **Confirm:** hash rows and check for overlap between the splits.
- **Fix:** split before any processing, deduplicate, and split by entity if needed.

### Validation label mismatch (`val_label_mismatch`)
- **Signature:** training is perfect while validation accuracy is near chance and validation loss climbs from epoch 1.
- **Commonly mistaken for:** catastrophic overfitting or shift. On 3 of 10 fresh seeds the val accuracy was 20–28%, not 10%: a random relabelling of 10 classes leaves about one class mapped to itself *on average*. A "below 20%" threshold missed it, and the diagnostician called it a shift.
- **Confirm:** the confusion matrix on validation is a near-permutation (most mass on one off-diagonal cell per row).
- **Fix:** use one label encoder fitted once and shared across splits, and save it with the model.
