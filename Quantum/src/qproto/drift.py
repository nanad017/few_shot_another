"""Concept drift detection module (paper, "Concept drift detection module").

The training data is partitioned chronologically into T = 6 equal-sized splits.
For each evaluation round t in {1, ..., T-1}, splits 1..t form the cumulative
training set, split t+1 is the hold-out test set; a fresh prototypical network
is trained for 100 episodes and evaluated over 50 episodes. Drift magnitude is
the relative degradation from the initial baseline,
    Delta^(t) = (A^(1) - A^(t)) / A^(1) * 100,
and an alert fires when Delta exceeds theta = 15%.
"""

import numpy as np

from .config import Config
from .train import train_protonet, episodic_accuracy


def drift_evaluation(X, y, timestamps, cfg: Config, device="cpu", verbose=True):
    """Run the cumulative temporal split protocol; returns per-round results."""
    order = np.argsort(timestamps)
    X, y = X[order], y[order]
    splits = np.array_split(np.arange(len(X)), cfg.drift_splits)

    results = []
    baseline = None
    for t in range(1, cfg.drift_splits):
        train_idx = np.concatenate(splits[:t])
        test_idx = splits[t]
        model, _ = train_protonet(X[train_idx], y[train_idx], cfg,
                                  episodes=cfg.drift_train_episodes,
                                  log_every=10 ** 9, device=device)
        acc = episodic_accuracy(model, X[test_idx], y[test_idx], cfg,
                                episodes=cfg.drift_eval_episodes, device=device)
        if baseline is None:
            baseline = acc
        delta = (baseline - acc) / baseline * 100.0
        alert = delta > cfg.drift_threshold
        results.append({"round": t, "accuracy": acc, "drift_magnitude": delta,
                        "alert": alert})
        if verbose:
            flag = "  << DRIFT ALERT" if alert else ""
            print(f"round {t}: train splits 1..{t}, test split {t + 1} -> "
                  f"acc {acc*100:.2f}%  Delta {delta:+.2f}%{flag}")
    return results
