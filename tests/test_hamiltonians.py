import numpy as np
import pennylane as qml
import pytest

from cluster_mbqc_toolkit import ClusterHamiltonian, RotatedHamiltonian

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def _dense_HC(N: int) -> np.ndarray:
    """Independent implementation of H_C via plain numpy Kronecker
    products -- used only to cross-check RotatedHamiltonian below, so a
    bug shared between the two would have to be a genuinely subtle one."""

    def op_on(site_ops):
        mats = [site_ops.get(i, I2) for i in range(N)]
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = np.zeros((2**N, 2**N), dtype=complex)
    H -= op_on({0: X, 1: Z})
    for j in range(1, N - 1):
        H -= op_on({j - 1: Z, j: X, j + 1: Z})
    H -= op_on({N - 2: Z, N - 1: X})
    return H


def _dense_RY_minus_theta_product(N: int, theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    ry_minus = np.array([[c, s], [-s, c]], dtype=complex)
    U = ry_minus
    for _ in range(N - 1):
        U = np.kron(U, ry_minus)
    return U


@pytest.mark.parametrize("N", [4, 6])
@pytest.mark.parametrize("theta", [0.0, 0.4, 1.0, np.pi / 2, 2.1])
def test_rotated_hamiltonian_matches_brute_force_conjugation(N, theta):
    """RotatedHamiltonian's analytic term-by-term expansion must equal a
    completely independent brute-force numpy matrix conjugation."""
    H_dense = _dense_HC(N)
    U = _dense_RY_minus_theta_product(N, theta)
    expected = U @ H_dense @ U.conj().T

    H = RotatedHamiltonian(ClusterHamiltonian(N), theta)
    actual = qml.matrix(H.to_qml_hamiltonian(), wire_order=range(N))

    np.testing.assert_allclose(actual, expected, atol=1e-9)


@pytest.mark.parametrize("N", [4, 6, 8])
@pytest.mark.parametrize("theta", [0.0, 0.4, 1.0, np.pi / 2, 2.1, 3.0])
def test_ground_energy_is_theta_independent(N, theta):
    """Conjugating H_C by a purely local unitary can never change its
    spectrum: the ground energy must be exactly -N for every theta, not
    just theta=0. This is the headline physics result, not a coincidence
    to work around."""
    H = RotatedHamiltonian(ClusterHamiltonian(N), theta)
    assert H.ground_energy_exact() == pytest.approx(-N, abs=1e-6)


def test_theta_zero_reproduces_the_base_hamiltonian():
    N = 5
    base = ClusterHamiltonian(N)
    rotated = RotatedHamiltonian(base, 0.0)
    np.testing.assert_allclose(
        qml.matrix(rotated.to_qml_hamiltonian(), wire_order=range(N)),
        qml.matrix(base.to_qml_hamiltonian(), wire_order=range(N)),
        atol=1e-10,
    )
