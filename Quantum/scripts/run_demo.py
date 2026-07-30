"""End-to-end demo of the full framework on synthetic data.

Runs all five modules of the paper: preprocessing, CatBoost feature selection,
episodic prototypical training, the hybrid quantum classifier, and concept
drift detection. Use scripts/run_experiment.py with real CSVs of
CCCS-CIC-AndMal-2020 / KronoDroid to reproduce the paper's numbers.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from qproto import Config, Preprocessor
from qproto.data import make_synthetic, stratified_split
from qproto.feature_selection import select_features
from qproto.train import train_protonet, episodic_accuracy, full_test_evaluation
from qproto.quantum_train import train_quantum_classifier, predict
from qproto.drift import drift_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--n-features", type=int, default=51)
    ap.add_argument("--quantum-epochs", type=int, default=20)
    args = ap.parse_args()

    cfg = Config()
    print("=== [A] Synthetic data (15 classes, drifting over time) ===")
    X, y, ts = make_synthetic(n_classes=15, n_per_class=400, d=200, seed=cfg.seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(X, y, seed=cfg.seed)
    print(f"train/val/test: {len(ytr)}/{len(yva)}/{len(yte)}, d={X.shape[1]}")

    prep = Preprocessor()
    Xtr_s, Xva_s, Xte_s = prep.fit_transform(Xtr), prep.transform(Xva), prep.transform(Xte)

    print(f"\n=== [B] CatBoost feature selection (top {args.n_features}) ===")
    sel = select_features(Xtr_s, ytr, Xva_s, yva, k=args.n_features, seed=cfg.seed)
    print(f"{X.shape[1]} -> {len(sel)} features "
          f"({(1 - len(sel)/X.shape[1])*100:.2f}% reduction)")
    Xtr_s, Xva_s, Xte_s = Xtr_s[:, sel], Xva_s[:, sel], Xte_s[:, sel]

    print(f"\n=== [C] Prototypical network, 5-way 5-shot, {args.episodes} episodes ===")
    model, _ = train_protonet(Xtr_s, ytr, cfg, X_val=Xva_s, y_val=yva,
                              episodes=args.episodes)
    acc = episodic_accuracy(model, Xte_s, yte, cfg, episodes=500)
    print(f"test 5-way 5-shot episodic accuracy: {acc*100:.2f}%")

    y_pred = full_test_evaluation(model, Xtr_s, ytr, Xte_s, yte, cfg)
    print(f"full test-set accuracy (5 support/class): "
          f"{accuracy_score(yte, y_pred)*100:.2f}%")
    print(classification_report(yte, y_pred, digits=3))

    print("=== [D] Hybrid quantum-classical classifier (4-qubit PQC) ===")
    qmodel = train_quantum_classifier(Xtr_s, ytr, Xva_s, yva,
                                      n_classes=len(np.unique(y)),
                                      epochs=args.quantum_epochs, seed=cfg.seed)
    q_acc = accuracy_score(yte, predict(qmodel, Xte_s))
    print(f"quantum pathway test accuracy: {q_acc*100:.2f}%")

    print("\n=== [E] Concept drift detection (T=6 cumulative splits) ===")
    X_all_s = prep.transform(X)[:, sel]
    drift_evaluation(X_all_s, y, ts, cfg)


if __name__ == "__main__":
    main()
