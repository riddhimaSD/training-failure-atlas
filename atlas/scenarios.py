"""The failure catalogue. Every scenario changes exactly one thing relative to the healthy run."""
from dataclasses import replace

from .core import HEALTHY

SCENARIOS = {
    "healthy": (HEALTHY, "Baseline: standardised inputs, shuffled mini-batches, SGD+momentum lr 0.05."),
    "lr_too_high": (replace(HEALTHY, lr=1.5), "Learning rate 30x too high."),
    "lr_too_low": (replace(HEALTHY, lr=2e-4), "Learning rate 250x too low."),
    "overfitting": (replace(HEALTHY, train_size=100, hidden=(512, 512), epochs=60),
                    "100 training examples, a wide network, no regularisation."),
    "label_noise": (replace(HEALTHY, label_noise=0.5), "Half of the training labels replaced at random."),
    "unshuffled": (replace(HEALTHY, shuffle=False), "Training data sorted by class and never shuffled."),
    "unnormalized_inputs": (replace(HEALTHY, standardize=False, input_scale=16.0),
                            "Raw pixel values scaled to 0..256, no standardisation."),
    "dead_relus": (replace(HEALTHY, hidden_bias=-4.0), "Hidden biases initialised to -4, so ReLUs start dead."),
    "vanishing_gradients": (replace(HEALTHY, hidden=(64,) * 8, activation="sigmoid"),
                            "8 sigmoid layers, no normalisation or residuals."),
    "exploding_init": (replace(HEALTHY, hidden=(128,) * 8, init_gain=5.0), "8 ReLU layers with weights initialised 5x too large."),
    "val_distribution_shift": (replace(HEALTHY, val_shift=3.0), "Validation inputs shifted by +3 std (a preprocessing mismatch)."),
    "val_leakage": (replace(HEALTHY, val_from_train=True), "Validation rows drawn from the training set."),
    "val_label_mismatch": (replace(HEALTHY, val_label_perm=True), "Validation labels use a different class-index mapping."),
}
