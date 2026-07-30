"""Quantum-enhanced hybrid classification layer (paper, "Quantum-enhanced
hybrid classification layer" and Fig. 1D).

Pipeline for input x in R^d (CatBoost-selected features):
    h_pre = ReLU(W_in x + b_in)          W_in: 64 x d
    z     = tanh(W_pre h_pre + b_pre)    W_pre: 4 x 64, z in [-1, 1]^4
    |psi> = U_var(z) U_ent U_enc(z) |0>^4   with
        U_enc(z) = tensor_i RY(z_i)                       (rotation encoding)
        U_ent    = CNOT(3,0) CNOT(2,3) CNOT(1,2) CNOT(0,1) (ring entanglement)
        U_var(z) = tensor_i RZ(alpha * z_i), alpha = 0.5   (scaled variational)
    o_i   = <psi| Z_i |psi>  in [-1, 1]   (Pauli-Z measurement per qubit)
    logits = post-quantum classical network(o)

The paper simulates the circuit with Qiskit; here the 4-qubit statevector is
simulated exactly in PyTorch, which is mathematically identical and gives exact
gradients through autograd (equal to parameter-shift values). As specified in
the paper, the rotation angles are functions of z only (the trainable
parameters live in the pre- and post-quantum classical networks). The model is
trained with cross-entropy loss.
"""

import torch
import torch.nn as nn


def _apply_1q(state: torch.Tensor, gate: torch.Tensor, qubit: int, n: int) -> torch.Tensor:
    """Apply a batched single-qubit gate (B,2,2) to state (B, 2^n)."""
    B = state.shape[0]
    s = state.reshape(B, *([2] * n)).movedim(1 + qubit, 1).reshape(B, 2, -1)
    s = torch.einsum("bij,bjk->bik", gate, s)
    return s.reshape(B, 2, *([2] * (n - 1))).movedim(1, 1 + qubit).reshape(B, -1)


def _apply_cnot(state: torch.Tensor, control: int, target: int, n: int) -> torch.Tensor:
    """Apply CNOT(control, target) to state (B, 2^n)."""
    B = state.shape[0]
    # move control axis to dim 1, target axis to dim 2
    s = state.reshape(B, *([2] * n)).movedim(1 + control, 1)
    t = 1 + target + (1 if target < control else 0)  # target axis after the first move
    s = s.movedim(t, 2).reshape(B, 2, 2, -1)
    s = torch.stack([s[:, 0, 0], s[:, 0, 1], s[:, 1, 1], s[:, 1, 0]], dim=1)
    s = s.reshape(B, 2, 2, *([2] * (n - 2))).movedim(2, t).movedim(1, 1 + control)
    return s.reshape(B, -1)


class QuantumCircuit(nn.Module):
    """Exact statevector simulation of the 4-qubit PQC U(z) = U_var U_ent U_enc."""

    def __init__(self, n_qubits: int = 4, alpha: float = 0.5):
        super().__init__()
        self.n = n_qubits
        self.alpha = alpha

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, n_qubits) in [-1,1] -> Pauli-Z expectations (B, n_qubits)."""
        B, n = z.shape
        dtype = torch.complex64
        state = torch.zeros(B, 2 ** n, dtype=dtype, device=z.device)
        state[:, 0] = 1.0  # |0...0>

        # Rotation encoding: RY(z_i) on each qubit
        half = (z / 2)
        c, s = torch.cos(half), torch.sin(half)
        for q in range(n):
            ry = torch.stack([
                torch.stack([c[:, q], -s[:, q]], dim=1),
                torch.stack([s[:, q], c[:, q]], dim=1)], dim=1).to(dtype)
            state = _apply_1q(state, ry, q, n)

        # Ring entanglement: CNOT(0,1), CNOT(1,2), CNOT(2,3), CNOT(3,0)
        for q in range(n):
            state = _apply_cnot(state, q, (q + 1) % n, n)

        # Variational layer: RZ(alpha * z_i) on each qubit
        phase = self.alpha * z / 2
        for q in range(n):
            e_neg = torch.exp(-1j * phase[:, q].to(dtype))
            e_pos = torch.exp(1j * phase[:, q].to(dtype))
            zero = torch.zeros_like(e_neg)
            rz = torch.stack([
                torch.stack([e_neg, zero], dim=1),
                torch.stack([zero, e_pos], dim=1)], dim=1)
            state = _apply_1q(state, rz, q, n)

        # Pauli-Z expectation per qubit: sum of |amp|^2 weighted by (-1)^bit
        probs = state.abs().pow(2).reshape(B, *([2] * n))
        out = []
        for q in range(n):
            p = probs.movedim(1 + q, 1).reshape(B, 2, -1).sum(dim=2)
            out.append(p[:, 0] - p[:, 1])
        return torch.stack(out, dim=1)


class HybridQuantumClassifier(nn.Module):
    """Pre-quantum MLP -> 4-qubit PQC -> post-quantum classical head.

    Serves as the alternative classification pathway of Fig. 1D, operating on
    the same CatBoost-selected features and trained with cross-entropy.
    """

    def __init__(self, in_dim: int, n_classes: int, n_qubits: int = 4,
                 alpha: float = 0.5, post_hidden: int = 16):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                                 nn.Linear(64, n_qubits), nn.Tanh())
        self.circuit = QuantumCircuit(n_qubits, alpha)
        self.post = nn.Sequential(nn.Linear(n_qubits, post_hidden), nn.ReLU(),
                                  nn.Linear(post_hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.circuit(self.pre(x)))
