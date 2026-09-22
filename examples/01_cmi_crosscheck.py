"""
Rebuild of the original pennylane_cmi_crosscheck.py exploration, using the
package's classes instead of hand-written circuits: cross-checks the
conditional mutual information of the rotated cluster/ES state against the
thesis's exact endpoint result (Eq. 71-72), then compares the exact
statevector against a bond-dimension-2 MPS device (Eq. 58: cluster states
are EXACTLY D=2 MPS, so the fidelity should be 1).

Run: python examples/01_cmi_crosscheck.py
"""
import numpy as np

from cluster_mbqc_toolkit import (
    ClusterStateCircuit,
    EntanglementSwappedCircuit,
    RotatedCircuit,
    CMIAnalyzer,
    make_device,
)

if __name__ == "__main__":
    print("=== CMI cross-check against thesis Eq. 71-72 (ES protocol) ===")
    for N in (6, 8, 10, 12):
        base = EntanglementSwappedCircuit(N)
        cmi0 = CMIAnalyzer(RotatedCircuit(base, 0.0).probs(), N).endpoint_cmi()
        cmi_pi2 = CMIAnalyzer(RotatedCircuit(base, np.pi / 2).probs(), N).endpoint_cmi()
        print(f"N={N:2d}  CMI(theta=0) = {cmi0:.6f} bits (expect 1)   "
              f"CMI(theta=pi/2) = {cmi_pi2:.2e} bits (expect 0)")

    print("\n=== default.tensor (MPS, chi=2) vs default.qubit: state fidelity ===")
    print("(requires `pip install quimb`; see README if this raises an error)")
    for N in (8, 12):
        base = EntanglementSwappedCircuit(N)
        circuit = RotatedCircuit(base, theta=0.35)

        exact_state = circuit.state()
        mps_dev = make_device("default.tensor", N, method="mps", max_bond_dim=2)
        mps_state = circuit.state(device=mps_dev)

        fidelity = float(np.abs(np.vdot(exact_state, mps_state)) ** 2)
        print(f"N={N:2d}  fidelity = {fidelity:.12f}  (bond dim 2 is EXACT for cluster states, Eq. 58)")
