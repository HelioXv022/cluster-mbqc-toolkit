"""
State-preparation circuits for 1D cluster states, thesis Eq. 4-9, 67-73.

Class design
------------
``StatePreparationCircuit`` is the interface: anything that can append its
gates to whatever QNode is active. It also gives every subclass working
``probs()`` / ``state()`` convenience methods for free (a template method
built only out of the abstract ``build()``), so a new circuit only ever has
to implement ``build()``.

``EntanglementSwappedCircuit`` is an inheritance example: the 'ES' readout
protocol IS a cluster-state preparation plus one more Hadamard layer, so it
subclasses ``ClusterStateCircuit`` and extends ``build()`` with
``super().build()``.

``RotatedCircuit`` is deliberately NOT inheritance: applying a per-site
RY(-theta) layer is an orthogonal piece of behaviour that should work on
top of *any* ``StatePreparationCircuit`` (cluster, ES, or something added
later), so it wraps a circuit instance instead of subclassing one --
composition over inheritance, chosen because the added behaviour doesn't
depend on which concrete circuit it's decorating.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pennylane as qml


class StatePreparationCircuit(ABC):
    """Interface for a state-preparation circuit on ``num_wires`` qubits."""

    def __init__(self, num_wires: int):
        if num_wires < 2:
            raise ValueError("need at least 2 wires")
        self._num_wires = num_wires

    @property
    def num_wires(self) -> int:
        return self._num_wires

    @property
    def wires(self):
        return range(self._num_wires)

    @abstractmethod
    def build(self) -> None:
        """Append this circuit's gates to the currently-recording QNode."""
        raise NotImplementedError

    def probs(self, device: Optional["qml.Device"] = None):
        """Measurement-basis probability table, qml.probs bit ordering."""
        dev = device or qml.device("default.qubit", wires=self.num_wires)

        @qml.qnode(dev)
        def circuit():
            self.build()
            return qml.probs(wires=list(self.wires))

        return circuit()

    def state(self, device: Optional["qml.Device"] = None):
        """Exact statevector."""
        dev = device or qml.device("default.qubit", wires=self.num_wires)

        @qml.qnode(dev)
        def circuit():
            self.build()
            return qml.state()

        return circuit()


class ClusterStateCircuit(StatePreparationCircuit):
    """Plain 1D cluster state ('C' protocol, thesis Eq. 73): Hadamard on
    every site, then a CZ chain along the open 1D graph."""

    def build(self) -> None:
        for i in self.wires:
            qml.Hadamard(wires=i)
        for i in range(self.num_wires - 1):
            qml.CZ(wires=[i, i + 1])


class EntanglementSwappedCircuit(ClusterStateCircuit):
    """'ES' protocol (thesis Eq. 6, 68): H_ES = U_H H_C U_H^dagger with
    U_H = H_1 ... H_{N-1} (Hadamard on the first N-1 sites only) --
    the cluster state plus one more Hadamard layer."""

    def build(self) -> None:
        super().build()
        for i in range(self.num_wires - 1):
            qml.Hadamard(wires=i)


class RotatedCircuit(StatePreparationCircuit):
    """Wraps any ``StatePreparationCircuit`` and appends a per-site
    RY(-theta) layer (thesis Eq. 8-9) before measurement."""

    def __init__(self, base: StatePreparationCircuit, theta: float):
        super().__init__(base.num_wires)
        self._base = base
        self._theta = theta

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def base(self) -> StatePreparationCircuit:
        return self._base

    def build(self) -> None:
        self._base.build()
        for i in self.wires:
            qml.RY(-self._theta, wires=i)
