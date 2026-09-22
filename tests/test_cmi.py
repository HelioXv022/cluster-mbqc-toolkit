import numpy as np
import pytest

from cluster_mbqc_toolkit import CMIAnalyzer


def test_rejects_wrong_length_probability_table():
    with pytest.raises(ValueError):
        CMIAnalyzer(np.array([0.5, 0.5, 0.0]), num_wires=2)  # needs length 4


def test_product_distribution_has_zero_cmi():
    """Three independent uniform bits: I(A:C|B) must be exactly 0."""
    probs = np.full(8, 1 / 8)
    cmi = CMIAnalyzer(probs, num_wires=3).conditional_mutual_information(
        A=(0,), B=(1,), C=(2,)
    )
    assert cmi == pytest.approx(0.0, abs=1e-10)


def test_perfectly_correlated_endpoints_give_one_bit():
    """3 qubits, X_0 == X_2 always, X_1 uniform and independent of both:
    I(X_0:X_2|X_1) should be exactly 1 bit (all the endpoint correlation
    survives conditioning on the untouched middle qubit)."""
    probs = np.zeros(8)
    # bitstrings (x0,x1,x2) with x0==x2, uniform over the 4 allowed patterns
    for x0 in (0, 1):
        for x1 in (0, 1):
            idx = x0 * 4 + x1 * 2 + x0
            probs[idx] = 1 / 4
    cmi = CMIAnalyzer(probs, num_wires=3).endpoint_cmi()
    assert cmi == pytest.approx(1.0, abs=1e-10)
