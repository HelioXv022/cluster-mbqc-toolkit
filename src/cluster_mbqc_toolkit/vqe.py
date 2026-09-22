"""
Ansatz interface + a Hamiltonian/ansatz-agnostic VQE solver.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .hamiltonians import StabilizerHamiltonian


class Ansatz(ABC):
    """Interface for a parameterized circuit family. ``VQESolver`` only
    ever calls ``param_shape`` and ``circuit``, so any ansatz -- hardware-
    efficient, a structured/physics-informed one added later, anything --
    plugs into the same solver without the solver needing to change."""

    @abstractmethod
    def param_shape(self, num_wires: int) -> Tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def circuit(self, weights, num_wires: int) -> None:
        """Append this ansatz's gates, parameterized by ``weights``."""
        raise NotImplementedError


class HardwareEfficientAnsatz(Ansatz):
    """``layers`` rounds of (RY, RZ) on every qubit, followed by a ring of
    CNOTs (linear chain + one wraparound edge -- the wraparound matters:
    a plain linear-chain entangler gets stuck short of the exact energy
    at larger N)."""

    def __init__(self, layers: int = 6):
        self._layers = layers

    @property
    def layers(self) -> int:
        return self._layers

    def param_shape(self, num_wires: int) -> Tuple[int, ...]:
        return (self._layers, num_wires, 2)

    def circuit(self, weights, num_wires: int) -> None:
        for l in range(self._layers):
            for i in range(num_wires):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[num_wires - 1, 0])


class VQESolver:
    """Minimizes <psi(weights)|H|psi(weights)> over an ``Ansatz``'s
    parameters using PennyLane's autodiff (backprop, on a statevector
    simulator) and the Adam optimizer, with optional multi-start random
    restarts.

    Takes the Hamiltonian and the ansatz as constructor arguments
    (dependency injection) rather than hardcoding either -- swapping
    ``ClusterHamiltonian`` for ``RotatedHamiltonian``, or the hardware-
    efficient ansatz for a structured one, never touches this class."""

    def __init__(
        self,
        hamiltonian: StabilizerHamiltonian,
        ansatz: Ansatz,
        device_name: str = "default.qubit",
        stepsize: float = 0.08,
    ):
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._num_wires = hamiltonian.num_wires
        self._device_name = device_name
        self._stepsize = stepsize
        self._qml_h = hamiltonian.to_qml_hamiltonian()

    @property
    def num_wires(self) -> int:
        return self._num_wires

    def _make_cost(self):
        dev = qml.device(self._device_name, wires=self._num_wires)
        num_wires = self._num_wires
        ansatz = self._ansatz
        H = self._qml_h

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def cost(weights):
            ansatz.circuit(weights, num_wires)
            return qml.expval(H)

        return cost

    def run(self, seed: int = 0, steps: int = 300, verbose: bool = False) -> float:
        cost = self._make_cost()
        rng = np.random.default_rng(seed)
        shape = self._ansatz.param_shape(self._num_wires)
        weights = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=self._stepsize)
        for it in range(steps):
            weights, e = opt.step_and_cost(cost, weights)
            if verbose and it % 50 == 0:
                print(f"    step {it:4d}  E = {e:.6f}")
        return float(cost(weights))

    def run_multistart(self, n_restarts: int = 3, **kwargs) -> float:
        energies = [self.run(seed=seed, **kwargs) for seed in range(n_restarts)]
        return min(energies)
