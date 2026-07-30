"""Sanity checks for the PyTorch statevector simulation of the 4-qubit PQC."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from qproto.quantum import QuantumCircuit, _apply_1q, _apply_cnot


def reference_circuit(z, alpha=0.5):
    """Dense-matrix reference: U_var U_ent U_enc |0000> and Z expectations."""
    n = 4
    I = np.eye(2, dtype=complex)

    def ry(t):
        return np.array([[np.cos(t / 2), -np.sin(t / 2)],
                         [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)

    def rz(t):
        return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])

    def kron_at(gate, q):
        ops = [I] * n
        ops[q] = gate
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out

    def cnot(c, t):
        dim = 2 ** n
        U = np.zeros((dim, dim), dtype=complex)
        for b in range(dim):
            bits = [(b >> (n - 1 - i)) & 1 for i in range(n)]
            if bits[c] == 1:
                bits[t] ^= 1
            b2 = sum(bit << (n - 1 - i) for i, bit in enumerate(bits))
            U[b2, b] = 1
        return U

    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1
    for q in range(n):
        state = kron_at(ry(z[q]), q) @ state
    for q in range(n):
        state = cnot(q, (q + 1) % n) @ state
    for q in range(n):
        state = kron_at(rz(alpha * z[q]), q) @ state

    Z = np.diag([1, -1]).astype(complex)
    return np.array([np.real(state.conj() @ kron_at(Z, q) @ state)
                     for q in range(n)])


def test_against_reference():
    rng = np.random.default_rng(0)
    circuit = QuantumCircuit(4, alpha=0.5)
    for _ in range(20):
        z = rng.uniform(-1, 1, size=4)
        expected = reference_circuit(z)
        got = circuit(torch.tensor(z[None], dtype=torch.float32))[0].numpy()
        assert np.allclose(got, expected, atol=1e-5), (got, expected)
    print("OK: statevector simulation matches dense-matrix reference (20 random inputs)")


def test_gradients_flow():
    circuit = QuantumCircuit(4)
    z = torch.tensor([[0.3, -0.7, 0.1, 0.9]], requires_grad=True)
    out = circuit(z).sum()
    out.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    print("OK: gradients flow through the circuit")


def test_norm_preserved():
    circuit = QuantumCircuit(4)
    z = torch.rand(8, 4) * 2 - 1
    out = circuit(z)
    assert out.shape == (8, 4)
    assert (out.abs() <= 1 + 1e-5).all()
    print("OK: Z expectations in [-1, 1], batched shape correct")


if __name__ == "__main__":
    test_against_reference()
    test_gradients_flow()
    test_norm_preserved()
