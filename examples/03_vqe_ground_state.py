"""
Rebuild of pennylane_vqe_cluster.py using VQESolver: finds the ground
energy of H_C via a hardware-efficient ansatz + Adam, compared against the
exact checkpoint E0=-N (thesis Eq. 7).

Run: python examples/03_vqe_ground_state.py
"""
from cluster_mbqc_toolkit import ClusterHamiltonian, HardwareEfficientAnsatz, VQESolver

if __name__ == "__main__":
    print("=== VQE for H_C, hardware-efficient ansatz ===")
    print("(exact E0 = -N, thesis Eq. 7)\n")
    for N in (4, 6, 8):
        solver = VQESolver(ClusterHamiltonian(N), HardwareEfficientAnsatz(layers=6))
        energy = solver.run_multistart(n_restarts=3, steps=300)
        E0 = -N
        print(f"N={N}: VQE energy = {energy:.6f}   exact E0 = {E0}   "
              f"error/site = {abs(energy - E0) / N:.3e}")
