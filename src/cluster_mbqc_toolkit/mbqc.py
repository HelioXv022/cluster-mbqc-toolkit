"""
A single-qubit MBQC gate-teleportation gadget (thesis Section 5.5 and
notebooks/cluster_state/mbqc.ipynb both point at this; neither builds it).
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from scipy.linalg import sqrtm


class MBQCGadget:
    """One edge of a cluster state (CZ) used as a one-way-computer gate:
    entangle an input qubit with an ancilla in |+>, measure the input in an
    equatorial basis, and apply a classically-conditioned Pauli-X
    correction -- deterministically leaving H . PhaseShift(-theta) applied
    to the input state on the ancilla wire.

    Encapsulates both the gadget circuit (``apply``) and its own
    correctness check (``verify``) as methods on the same object, so "does
    this actually do what it claims" travels with the class instead of
    living in a separate, easy-to-forget script.

    One instance = one fixed theta (the gate this gadget implements);
    input states are supplied per call.
    """

    def __init__(self, theta: float):
        self._theta = theta
        self._dev2 = qml.device("default.qubit", wires=2)
        self._dev1 = qml.device("default.qubit", wires=1)

    @property
    def theta(self) -> float:
        return self._theta

    def apply(self, phi0: float, phi1: float) -> np.ndarray:
        """Density matrix of the OUTPUT qubit, for input state
        RY(phi0) RZ(phi1) |0>."""
        theta = self._theta

        @qml.qnode(self._dev2)
        def circuit():
            qml.RY(phi0, wires=0)
            qml.RZ(phi1, wires=0)
            qml.Hadamard(wires=1)
            qml.CZ(wires=[0, 1])
            qml.PhaseShift(-theta, wires=0)
            qml.Hadamard(wires=0)
            m = qml.measure(wires=0)
            qml.cond(m, qml.PauliX)(wires=1)
            return qml.density_matrix(wires=1)

        return circuit()

    def direct_gate_reference(self, phi0: float, phi1: float) -> np.ndarray:
        """The gate-based circuit this gadget should reproduce exactly:
        H . PhaseShift(-theta) applied directly, no entanglement/measurement."""
        theta = self._theta

        @qml.qnode(self._dev1)
        def circuit():
            qml.RY(phi0, wires=0)
            qml.RZ(phi1, wires=0)
            qml.PhaseShift(-theta, wires=0)
            qml.Hadamard(wires=0)
            return qml.density_matrix(wires=0)

        return circuit()

    @staticmethod
    def density_matrix_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
        rho, sigma = np.array(rho), np.array(sigma)
        sq = sqrtm(rho)
        return float(np.real(np.trace(sqrtm(sq @ sigma @ sq))) ** 2)

    def fidelity_to_direct_gate(self, phi0: float, phi1: float) -> float:
        return self.density_matrix_fidelity(
            self.apply(phi0, phi1), self.direct_gate_reference(phi0, phi1)
        )

    def verify(self, n_trials: int = 20, seed: int = 0) -> float:
        """Minimum fidelity to the direct-gate reference over ``n_trials``
        random input states -- should be 1 to numerical precision."""
        rng = np.random.default_rng(seed)
        fids = [
            self.fidelity_to_direct_gate(*rng.uniform(0, 2 * np.pi, size=2))
            for _ in range(n_trials)
        ]
        return float(np.min(fids))
