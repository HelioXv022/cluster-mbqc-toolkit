"""
cluster_mbqc_toolkit
=====================

A small, deliberately object-oriented PennyLane toolkit for rotated 1D
cluster states, built while extending my M.Sc. thesis ("Conditional
Information Flow and Neural Representability of One-Dimensional Rotated
Cluster States") into working quantum-circuit code.

See the top-level README for the physics background and the design
rationale behind the class hierarchy. In short:

- ``circuits``      -- state-preparation circuits (interface + inheritance
                        + a decorator-style wrapper)
- ``hamiltonians``  -- stabilizer Hamiltonians (interface + inheritance +
                        a composition-based rotation wrapper)
- ``cmi``           -- conditional mutual information from a probability
                        table (encapsulation)
- ``mbqc``          -- a single-qubit MBQC teleportation gadget
- ``vqe``           -- ansatz interface + a Hamiltonian/ansatz-agnostic
                        VQE solver (dependency injection / strategy pattern)
- ``devices``       -- a thin device factory with clearer error messages
                        for the tensor-network backend's optional dependency
"""

from .circuits import ClusterStateCircuit, EntanglementSwappedCircuit, RotatedCircuit
from .hamiltonians import ClusterHamiltonian, RotatedHamiltonian
from .cmi import CMIAnalyzer
from .mbqc import MBQCGadget
from .vqe import HardwareEfficientAnsatz, VQESolver
from .devices import make_device

__all__ = [
    "ClusterStateCircuit",
    "EntanglementSwappedCircuit",
    "RotatedCircuit",
    "ClusterHamiltonian",
    "RotatedHamiltonian",
    "CMIAnalyzer",
    "MBQCGadget",
    "HardwareEfficientAnsatz",
    "VQESolver",
    "make_device",
]

__version__ = "0.1.0"
