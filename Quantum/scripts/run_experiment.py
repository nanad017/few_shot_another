"""Reproduce the paper's experiments on a real dataset CSV.

Examples:
  # CCCS-CIC-AndMal-2020 (family classification, 51 features)
  python scripts/run_experiment.py --csv andmal2020.csv --label-col Class \
      --n-features 51 --episodes 4000

  # KronoDroid real-device (binary classification, 29 features, drift eval)
  python scripts/run_experiment.py --csv kronodroid_real.csv --label-col Malware \
      --timestamp-col FirstSeen --n-features 29 --episodes 4000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from qproto import Config, Preprocessor
from qproto.data import load_csv, stratified_split
from qproto.feature_selection import select_features
from qproto.train import train_protonet, episodic_accuracy, full_test_evaluation
from qproto.drift import drift_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--timestamp-col", default=None)
    ap.add_argument("--drop-cols", nargs="*", default=[])
    ap.add_argument("--n-features", type=int, default=51)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--n-way", type=int, default=5)
    args = ap.parse_args()

    cfg = Config(episodes=args.episodes, n_way=args.n_way)
    X, y, ts = load_csv(args.csv, args.label_col, args.timestamp_col,
                        args.drop_cols)
    # binary tasks (e.g. KronoDroid) cap N-way at the number of classes
    cfg.n_way = min(cfg.n_way, len(np.unique(y)))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(X, y, seed=cfg.seed)

    prep = Preprocessor()
    Xtr_s, Xva_s, Xte_s = prep.fit_transform(Xtr), prep.transform(Xva), prep.transform(Xte)

    print(f"CatBoost feature selection: {X.shape[1]} -> {args.n_features}")
    sel = select_features(Xtr_s, ytr, Xva_s, yva, k=args.n_features,
                          seed=cfg.seed, verbose=True)
    Xtr_s, Xva_s, Xte_s = Xtr_s[:, sel], Xva_s[:, sel], Xte_s[:, sel]

    model, _ = train_protonet(Xtr_s, ytr, cfg, X_val=Xva_s, y_val=yva)
    acc = episodic_accuracy(model, Xte_s, yte, cfg, episodes=500)
    print(f"test {cfg.n_way}-way 5-shot episodic accuracy: {acc*100:.2f}%")

    y_pred = full_test_evaluation(model, Xtr_s, ytr, Xte_s, yte, cfg)
    print(f"full test-set accuracy: {accuracy_score(yte, y_pred)*100:.2f}%")
    print(classification_report(yte, y_pred, digits=4))

    if ts is not None:
        print("Concept drift evaluation (T=6 cumulative temporal splits)")
        drift_evaluation(prep.transform(X)[:, sel], y, ts, cfg)


if __name__ == "__main__":
    main()
