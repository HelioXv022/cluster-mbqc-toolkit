import numpy as np
import pytest

from cluster_mbqc_toolkit import MBQCGadget


@pytest.mark.parametrize("theta", [0.0, 0.37, np.pi / 4, 1.9, 2 * np.pi - 0.1])
def test_gadget_matches_direct_gate_to_high_fidelity(theta):
    gadget = MBQCGadget(theta)
    assert gadget.verify(n_trials=10) == pytest.approx(1.0, abs=1e-6)


def test_single_trial_fidelity_directly():
    gadget = MBQCGadget(theta=0.9)
    fid = gadget.fidelity_to_direct_gate(phi0=1.1, phi1=0.4)
    assert fid == pytest.approx(1.0, abs=1e-8)
