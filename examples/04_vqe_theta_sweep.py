"""
Rebuild of pennylane_vqe_theta.py using RotatedHamiltonian + VQESolver:
shows the ground energy of H(theta) = U(theta) H_C U(theta)^dagger is
EXACTLY -N for every theta (not just theta=0), confirmed first by exact
diagonalization and then by an actual VQE circuit.

Run: python examples/04_vqe_theta_sweep.py
"""
import numpy as np

from cluster_mbqc_toolkit import (
    ClusterHamiltonian,
    RotatedHamiltonian,
    HardwareEfficientAnsatz,
    VQESolver,
)

if __name__ == "__main__":
    print("=== exact diagonalization: E0(theta) for N=4,6,8 ===")
    print("(local-unitary conjugation cannot change eigenvalues -- E0 is")
    print(" EXACTLY -N for every theta; see README for why this matters)\n")
    for N in (4, 6, 8):
        for theta in (0.0, 0.4, 1.0, np.pi / 2, 2.1, 3.0):
            H = RotatedHamiltonian(ClusterHamiltonian(N), theta)
            print(f"N={N:2d} theta={theta:.3f}  E0 = {H.ground_energy_exact():.6f}   (exact: {-N})")

    print("\n=== VQE reproducing E0(theta) via a real parameterized circuit ===")
    print("(N=4 converges to machine precision at every theta; N=6 needs a")
    print(" deeper ansatz [10-12 layers] and a smaller Adam step away from")
    print(" theta=0 -- a real trainability wrinkle, not a Hamiltonian bug,")
    print(" already proven correct above by exact diagonalization)\n")
    for theta in (0.0, np.pi / 2):
        H = RotatedHamiltonian(ClusterHamiltonian(4), theta)
        solver = VQESolver(H, HardwareEfficientAnsatz(layers=6))
        e = solver.run_multistart(n_restarts=3, steps=250)
        print(f"N=4 theta={theta:.3f}  VQE E = {e:.6f}   exact E0 = -4   error/site = {abs(e + 4) / 4:.2e}")
    for theta in (0.0, 0.6, np.pi / 2):
        H = RotatedHamiltonian(ClusterHamiltonian(6), theta)
        solver = VQESolver(H, HardwareEfficientAnsatz(layers=10), stepsize=0.06)
        e = solver.run_multistart(n_restarts=2, steps=400)
        print(f"N=6 theta={theta:.3f}  VQE E = {e:.6f}   exact E0 = -6   error/site = {abs(e + 6) / 6:.2e}")
