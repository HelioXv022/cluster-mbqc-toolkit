import numpy as np
import pytest

from cluster_mbqc_toolkit import (
    ClusterStateCircuit,
    EntanglementSwappedCircuit,
    RotatedCircuit,
    CMIAnalyzer,
)


def test_entanglement_swapped_is_a_cluster_state_circuit():
    """Inheritance check: ES protocol IS-A cluster-state circuit plus more."""
    circuit = EntanglementSwappedCircuit(4)
    assert isinstance(circuit, ClusterStateCircuit)


@pytest.mark.parametrize("N", [6, 8, 10, 12])
def test_endpoint_cmi_matches_thesis_eq_71_72(N):
    """theta=0 -> exactly 1 bit; theta=pi/2 -> exactly 0 bits (ES protocol)."""
    cmi_0 = CMIAnalyzer(RotatedCircuit(EntanglementSwappedCircuit(N), 0.0).probs(), N).endpoint_cmi()
    cmi_pi2 = CMIAnalyzer(
        RotatedCircuit(EntanglementSwappedCircuit(N), np.pi / 2).probs(), N
    ).endpoint_cmi()
    assert cmi_0 == pytest.approx(1.0, abs=1e-8)
    assert cmi_pi2 == pytest.approx(0.0, abs=1e-8)


def test_rotated_circuit_wraps_any_base_circuit():
    """Composition check: RotatedCircuit works on both concrete circuits
    without needing its own subclass per base."""
    for base_cls in (ClusterStateCircuit, EntanglementSwappedCircuit):
        wrapped = RotatedCircuit(base_cls(4), theta=0.3)
        probs = wrapped.probs()
        assert probs.shape == (16,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-10)


def test_theta_zero_rotation_is_identity_on_cluster_state():
    """RotatedCircuit(base, 0.0) should reproduce base's own probabilities."""
    base = ClusterStateCircuit(5)
    rotated = RotatedCircuit(base, 0.0)
    np.testing.assert_allclose(base.probs(), rotated.probs(), atol=1e-10)
