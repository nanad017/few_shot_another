"""Hyperparameters from Table 3 of the paper."""

from dataclasses import dataclass


@dataclass
class Config:
    # Few-shot learning
    n_way: int = 5
    k_shot: int = 5
    q_query: int = 10
    episodes: int = 4000
    embedding_dim: int = 128
    dropout: float = 0.3

    # Embedding network
    hidden1: int = 512
    hidden2: int = 256

    # Quantum circuit
    n_qubits: int = 4
    alpha: float = 0.5  # rotation scaling for variational R_Z layer

    # Optimization
    lr: float = 1e-3
    lr_step: int = 1000
    lr_gamma: float = 0.5

    # Feature selection
    n_features_cic: int = 51    # CCCS-CIC-AndMal-2020: 9,503 -> 51
    n_features_krono: int = 29  # KronoDroid: 489 -> 29

    # Drift detection
    drift_splits: int = 6
    drift_train_episodes: int = 100
    drift_eval_episodes: int = 50
    drift_threshold: float = 15.0  # percent degradation triggering alert

    seed: int = 42
