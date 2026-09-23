"""
How to plug a NEW Hamiltonian and a NEW circuit into this package without
touching any of its source files.

The new model: the GHZ stabilizer Hamiltonian on an open chain,

    H_GHZ = - sum_{i=0}^{N-2} Z_i Z_{i+1}  -  X_0 X_1 ... X_{N-1}

Its N terms commute and are the stabilizers of the GHZ state
(|00...0> + |11...1>) / sqrt(2), so the ground energy is exactly -N.

Step 1: subclass StabilizerHamiltonian, implement terms().
Step 2: subclass StatePreparationCircuit, implement build().
Step 3: use them with every existing tool -- ground_energy_exact(),
        RotatedHamiltonian, RotatedCircuit, VQESolver, CMIAnalyzer --
        none of which were written with GHZ in mind.

Run: python examples/06_plug_in_new_hamiltonian.py
"""
import numpy as np
import pennylane as qml

from cluster_mbqc_toolkit import (
    CMIAnalyzer,
    ClusterStateCircuit,
    EntanglementSwappedCircuit,
    HardwareEfficientAnsatz,
    RotatedCircuit,
    RotatedHamiltonian,
    VQESolver,
)
from cluster_mbqc_toolkit.circuits import StatePreparationCircuit
from cluster_mbqc_toolkit.hamiltonians import StabilizerHamiltonian


# ---------------------------------------------------------------- step 1
class GHZHamiltonian(StabilizerHamiltonian):
    """H_GHZ = -sum Z_i Z_{i+1} - X^{(x)N}. Only terms() is new."""

    def terms(self):
        N = self.num_wires
        terms = [(-1.0, qml.PauliZ(i) @ qml.PauliZ(i + 1)) for i in range(N - 1)]
        terms.append((-1.0, qml.prod(*[qml.PauliX(i) for i in range(N)])))
        return terms


# ---------------------------------------------------------------- step 2
class GHZStateCircuit(StatePreparationCircuit):
    """Hadamard on wire 0, then a CNOT chain. Only build() is new."""

    def build(self):
        qml.Hadamard(wires=0)
        for i in range(self.num_wires - 1):
            qml.CNOT(wires=[i, i + 1])


def energy(hamiltonian, circuit):
    """<psi|H|psi> for a circuit's exact state and a Hamiltonian's matrix."""
    psi = circuit.state()
    return float(np.real(np.vdot(psi, hamiltonian.matrix() @ psi)))


# ---------------------------------------------------------------- step 3
if __name__ == "__main__":
    N = 6

    print("=== 1. Inherited for free: exact diagonalization ===")
    H = GHZHamiltonian(N)
    print(f"GHZHamiltonian(N={N}).ground_energy_exact() = {H.ground_energy_exact():.6f}  (exact: {-N})")

    print("\n=== 2. The new circuit prepares the new Hamiltonian's ground state ===")
    print(f"<GHZ| H_GHZ |GHZ> = {energy(H, GHZStateCircuit(N)):.6f}  (exact: {-N})")

    print("\n=== 3. Composition for free: RotatedHamiltonian / RotatedCircuit wrap GHZ ===")
    for theta in (0.0, 0.7, np.pi / 2):
        H_rot = RotatedHamiltonian(GHZHamiltonian(N), theta)
        psi_rot = RotatedCircuit(GHZStateCircuit(N), theta)
        print(f"theta={theta:.3f}  E0 = {H_rot.ground_energy_exact():.6f}   "
              f"<psi(theta)|H(theta)|psi(theta)> = {energy(H_rot, psi_rot):.6f}")

    print("\n=== 4. Dependency injection for free: VQESolver on the new Hamiltonian ===")
    solver = VQESolver(GHZHamiltonian(4), HardwareEfficientAnsatz(layers=6))
    print(f"VQE on GHZHamiltonian(N=4): E = {solver.run_multistart(n_restarts=2, steps=250):.6f}  (exact: -4)")

    print("\n=== 5. Physics: correlation vs. conditional mutual information ===")
    ghz = CMIAnalyzer(GHZStateCircuit(N).probs(), N)
    es = CMIAnalyzer(EntanglementSwappedCircuit(N).probs(), N)
    for name, a in (("GHZ", ghz), ("ES cluster", es)):
        mi = a.conditional_mutual_information((0,), (), (N - 1,))
        print(f"{name:10s}  I(X_1:X_N) = {mi:.3f} bits   "
              f"I(X_1:X_N | middle) = {a.endpoint_cmi():.3f} bits")
    print("GHZ: the endpoints are perfectly correlated, but the middle qubits")
    print("already determine them, so the CMI is 0. The ES cluster state is the")
    print("opposite: no endpoint correlation on its own, 1 bit once the middle")
    print("is known. Correlation and CMI measure different things.")
