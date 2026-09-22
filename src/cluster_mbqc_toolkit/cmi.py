"""
Conditional mutual information from a measurement probability table,
thesis Eq. 10-18 (definition), Eq. 71-72 (the exact endpoint result this
toolkit's tests verify against).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class CMIAnalyzer:
    """Wraps a length-2**N probability table (``qml.probs`` bit ordering)
    and exposes the marginal-entropy machinery needed for conditional
    mutual information. The reshape into an N-axis tensor happens once in
    ``__init__`` -- callers never touch the raw array's indexing
    convention (encapsulation)."""

    def __init__(self, probs: Sequence[float], num_wires: int):
        probs = np.asarray(probs)
        if probs.shape != (2 ** num_wires,):
            raise ValueError(f"expected a length-{2**num_wires} probability table")
        self._num_wires = num_wires
        self._probs = probs.reshape((2,) * num_wires)  # axis i = qubit i

    @property
    def num_wires(self) -> int:
        return self._num_wires

    def entropy_of_marginal(self, keep_axes: Sequence[int]) -> float:
        """Shannon entropy, in bits, of the marginal over ``keep_axes``."""
        drop_axes = tuple(a for a in range(self._num_wires) if a not in keep_axes)
        marg = self._probs.sum(axis=drop_axes) if drop_axes else self._probs
        marg = marg.flatten()
        marg = marg[marg > 1e-14]
        return float(-(marg * np.log2(marg)).sum())

    def conditional_mutual_information(
        self, A: Sequence[int], B: Sequence[int], C: Sequence[int]
    ) -> float:
        """I(A:C|B) in bits, thesis Eq. 12-13:
            I(A:C|B) = H(AB) + H(BC) - H(B) - H(ABC)."""
        A, B, C = tuple(A), tuple(B), tuple(C)
        H_AB = self.entropy_of_marginal(A + B)
        H_BC = self.entropy_of_marginal(B + C)
        H_B = self.entropy_of_marginal(B)
        H_ABC = self.entropy_of_marginal(A + B + C)
        return H_AB + H_BC - H_B - H_ABC

    def endpoint_cmi(self) -> float:
        """I(X_1 : X_N | X_2,...,X_{N-1}) in bits -- the endpoint quantity
        thesis Eq. 71-72 gives an exact answer for (1 bit at theta=0, 0 at
        theta=pi/2)."""
        N = self._num_wires
        A, C, B = (0,), (N - 1,), tuple(range(1, N - 1))
        return self.conditional_mutual_information(A, B, C)
