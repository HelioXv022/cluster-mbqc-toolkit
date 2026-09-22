"""
Stabilizer Hamiltonians for the rotated cluster-state family, thesis Eq. 5-9.

Class design
------------
``StabilizerHamiltonian`` is the interface: a subclass only has to provide
``terms()`` (a list of (coefficient, Pauli-string) pairs); the base class
turns that into a working ``qml.Hamiltonian``, its dense matrix, and an
exact-diagonalization ground energy -- a template-method pattern so that
adding a new Hamiltonian never means re-deriving "how do I build a
qml.Hamiltonian" or "how do I check its ground energy".

``RotatedHamiltonian`` WRAPS a ``StabilizerHamiltonian`` rather than
subclassing ``ClusterHamiltonian`` directly, so the same rotation logic
works on any stabilizer Hamiltonian someone adds later -- mirrors
``RotatedCircuit`` in circuits.py (composition, not inheritance, because
"apply a local rotation" is orthogonal to "which stabilizer Hamiltonian").
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pennylane as qml
from pennylane.operation import Operator


class StabilizerHamiltonian(ABC):
    """Interface for H = sum_i coeff_i * P_i, P_i a Pauli-string operator."""

    def __init__(self, num_wires: int):
        self._num_wires = num_wires

    @property
    def num_wires(self) -> int:
        return self._num_wires

    @abstractmethod
    def terms(self) -> List[Tuple[float, Operator]]:
        raise NotImplementedError

    def to_qml_hamiltonian(self) -> qml.Hamiltonian:
        coeffs, obs = zip(*self.terms())
        return qml.Hamiltonian(list(coeffs), list(obs))

    def matrix(self) -> np.ndarray:
        return qml.matrix(self.to_qml_hamiltonian(), wire_order=range(self.num_wires))

    def ground_energy_exact(self) -> float:
        """Exact diagonalization -- a ground-truth check for VQE, used in
        tests up to ~10-12 wires. Never intended to replace VQE at scale."""
        return float(np.linalg.eigvalsh(self.matrix())[0])


class ClusterHamiltonian(StabilizerHamiltonian):
    """H_C, thesis Eq. 5, open 1D chain:

        H_C = -X_0 Z_1 - sum_{j=1}^{N-2} Z_{j-1} X_j Z_{j+1} - Z_{N-2} X_{N-1}

    Every term is a Pauli string that squares to the identity, and all
    terms mutually commute (they're the cluster state's stabilizer group up
    to sign) -- an exactly solvable, frustration-free model with ground
    energy exactly -N (thesis Eq. 7)."""

    def terms(self) -> List[Tuple[float, Operator]]:
        N = self.num_wires
        terms: List[Tuple[float, Operator]] = [(-1.0, qml.PauliX(0) @ qml.PauliZ(1))]
        for j in range(1, N - 1):
            terms.append((-1.0, qml.PauliZ(j - 1) @ qml.PauliX(j) @ qml.PauliZ(j + 1)))
        terms.append((-1.0, qml.PauliZ(N - 2) @ qml.PauliX(N - 1)))
        return terms


class RotatedHamiltonian(StabilizerHamiltonian):
    """H(theta) = U(theta) H_base U(theta)^dagger, U(theta) = RY(-theta)^{⊗N}
    -- conjugates every term of a wrapped ``StabilizerHamiltonian`` by the
    same per-site rotation ``RotatedCircuit`` applies to the state, so that
    H(theta)'s ground state is exactly what that circuit prepares.

    Single-qubit conjugation rule (U = RY(-theta) per site), verified
    against a brute-force numpy matrix conjugation in
    tests/test_hamiltonians.py:
        U X U^dagger =  cos(theta) X + sin(theta) Z
        U Z U^dagger = -sin(theta) X + cos(theta) Z
    A k-site term expands into up to 2**k Pauli strings after substituting
    each site's letter and multiplying out.

    Physics note: conjugating by a purely LOCAL unitary can never change a
    Hamiltonian's eigenvalues, and it preserves the mutually-commuting
    stabilizer structure -- so ``ground_energy_exact()`` returns EXACTLY
    -N for every theta, not just theta=0. See the top-level README for why
    that's the actual point (energy is theta-independent; the measurement
    correlations captured by CMIAnalyzer are not).
    """

    _LETTER_OF = {qml.PauliX: "X", qml.PauliZ: "Z"}
    _CLASS_OF = {"X": qml.PauliX, "Z": qml.PauliZ}

    def __init__(self, base: StabilizerHamiltonian, theta: float):
        super().__init__(base.num_wires)
        self._base = base
        self._theta = theta

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def base(self) -> StabilizerHamiltonian:
        return self._base

    @classmethod
    def _term_to_site_letters(cls, observable: Operator):
        factors = observable.operands if hasattr(observable, "operands") else [observable]
        return [(op.wires[0], cls._LETTER_OF[type(op)]) for op in factors]

    def _rotate_site_letter(self, letter: str):
        c, s = np.cos(self._theta), np.sin(self._theta)
        if letter == "X":
            return [(c, "X"), (s, "Z")]
        return [(-s, "X"), (c, "Z")]

    def _rotate_term(self, sites_letters):
        expansions = [self._rotate_site_letter(letter) for _, letter in sites_letters]
        wires = [w for w, _ in sites_letters]
        out = [(1.0, {})]
        for wire, choices in zip(wires, expansions):
            out = [
                (coeff_so_far * c, {**letters_so_far, wire: letter})
                for coeff_so_far, letters_so_far in out
                for c, letter in choices
            ]
        return out

    def terms(self) -> List[Tuple[float, Operator]]:
        result: List[Tuple[float, Operator]] = []
        for base_coeff, base_obs in self._base.terms():
            sites_letters = self._term_to_site_letters(base_obs)
            for coeff, letters in self._rotate_term(sites_letters):
                op = None
                for wire in sorted(letters):
                    factor = self._CLASS_OF[letters[wire]](wire)
                    op = factor if op is None else op @ factor
                result.append((base_coeff * coeff, op))
        return result
