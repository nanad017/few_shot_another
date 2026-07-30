"""CatBoost-based feature selection (Algorithm 1, Eq. 2).

A CatBoost classifier is trained on the training set with default
hyperparameters; features are ranked by permutation importance measured as the
loss increase on the validation set, and the top-k indices are returned
(k = 51 for CCCS-CIC-AndMal-2020, k = 29 for KronoDroid).

`method="loss_change"` uses CatBoost's built-in LossFunctionChange importance
(fast, recommended for very high-dimensional data such as d = 9,503).
`method="permutation"` implements Eq. 2 literally with M random permutations
per feature.
"""

import numpy as np
from catboost import CatBoostClassifier, Pool


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k: int,
    method: str = "loss_change",
    n_permutations: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray:
    """Return indices of the top-k features, sorted by importance (descending)."""
    model = CatBoostClassifier(random_seed=seed, verbose=100 if verbose else 0)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    if method == "loss_change":
        val_pool = Pool(X_val, y_val)
        importance = model.get_feature_importance(data=val_pool, type="LossFunctionChange")
    elif method == "permutation":
        importance = _permutation_importance(model, X_val, y_val, n_permutations, seed)
    else:
        raise ValueError(f"unknown method: {method}")

    return np.argsort(importance)[::-1][:k]


def _permutation_importance(model, X_val, y_val, M: int, seed: int) -> np.ndarray:
    """Eq. 2: I_j = mean over M permutations of L(perm_j(D_val)) - L(D_val)."""
    rng = np.random.default_rng(seed)
    base_loss = _log_loss(model, X_val, y_val)
    d = X_val.shape[1]
    importance = np.zeros(d)
    for j in range(d):
        for _ in range(M):
            X_perm = X_val.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            importance[j] += _log_loss(model, X_perm, y_val) - base_loss
        importance[j] /= M
    return importance


def _log_loss(model, X, y) -> float:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    idx = np.array([classes.index(c) for c in y])
    p = np.clip(proba[np.arange(len(y)), idx], 1e-12, 1.0)
    return float(-np.log(p).mean())
