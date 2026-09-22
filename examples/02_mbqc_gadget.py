"""
Rebuild of pennylane_mbqc_gadget.py using MBQCGadget: verifies the
single-qubit gate-teleportation gadget against a direct gate-based
reference over random input states and angles.

Run: python examples/02_mbqc_gadget.py
"""
import numpy as np

from cluster_mbqc_toolkit import MBQCGadget

if __name__ == "__main__":
    print("=== MBQC single-qubit gate teleportation: gadget vs. direct gate ===")
    rng = np.random.default_rng(0)
    thetas = rng.uniform(0, 2 * np.pi, size=5)
    for theta in thetas:
        gadget = MBQCGadget(theta)
        fid = gadget.verify(n_trials=10)
        print(f"theta={theta:.4f}  min fidelity over 10 random input states = {fid:.10f}")
    print("(should all be 1 to numerical precision)")
