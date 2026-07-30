"""Data preprocessing pipeline (paper, "Data preprocessing pipeline").

Zero-imputation for missing values (Eq. before Eq. 1) and z-score
standardization with mean/std computed exclusively on the training set (Eq. 1),
then applied unchanged to validation and test sets.
"""

import numpy as np

EPS = 1e-8


class Preprocessor:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X_train: np.ndarray) -> "Preprocessor":
        X = np.nan_to_num(X_train, nan=0.0)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None:
            raise RuntimeError("Preprocessor must be fit on the training set first")
        X = np.nan_to_num(X, nan=0.0)
        return (X - self.mu) / (self.sigma + EPS)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)
