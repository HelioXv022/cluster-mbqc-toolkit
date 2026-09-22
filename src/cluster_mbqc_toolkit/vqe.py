"""
Ansatz interface + a Hamiltonian/ansatz-agnostic VQE solver.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .hamiltonians import StabilizerHamiltonian


class Ansatz(ABC):
    """Interface for a parameterized circuit family. ``VQESolver`` only
    ever calls ``param_shape`` and ``circuit``, so any ansatz -- hardware-
    efficient, a structured/physics-informed one added later, anything --
    plugs into the same solver without the solver needing to change."""

    @abstractmethod
    def param_shape(self, num_wires: int) -> Tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def circuit(self, weights, num_wires: int) -> None:
        """Append this ansatz's gates, parameterized by ``weights``."""
        raise NotImplementedError


class HardwareEfficientAnsatz(Ansatz):
    """``layers`` rounds of (RY, RZ) on every qubit, followed by a ring of
    CNOTs (linear chain + one wraparound edge -- the wraparound matters:
    a plain linear-chain entangler gets stuck short of the exact energy
    at larger N)."""

    def __init__(self, layers: int = 6):
        self._layers = layers

    @property
    def layers(self) -> int:
        return self._layers

    def param_shape(self, num_wires: int) -> Tuple[int, ...]:
        return (self._layers, num_wires, 2)

    def circuit(self, weights, num_wires: int) -> None:
        for l in range(self._layers):
            for i in range(num_wires):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[num_wires - 1, 0])


class VQESolver:
    """Minimizes <psi(weights)|H|psi(weights)> over an ``Ansatz``'s
    parameters using PennyLane's autodiff and the Adam optimizer, with
    optional multi-start random restarts.

    Takes the Hamiltonian and the ansatz as constructor arguments
    (dependency injection) rather than hardcoding either -- swapping
    ``ClusterHamiltonian`` for ``RotatedHamiltonian``, or the hardware-
    efficient ansatz for a structured one, never touches this class.

    ``interface`` selects the autodiff/optimization backend without
    changing anything else about the solver's public API:

    - ``"autograd"`` (default): PennyLane's own ``pennylane.numpy`` +
      ``qml.AdamOptimizer``.
    - ``"jax"``: the QNode is built with ``interface="jax"`` and wrapped
      in ``jax.jit``; the gradient step is ``jax.grad`` composed with a
      hand-rolled, ``jax.jit``-compiled Adam update (no PennyLane
      optimizer involved, since those expect ``pennylane.numpy`` arrays)
      -- requires the optional ``jax``/``jaxlib`` dependencies
      (``pip install -e ".[jax]"``).
    """

    def __init__(
        self,
        hamiltonian: StabilizerHamiltonian,
        ansatz: Ansatz,
        device_name: str = "default.qubit",
        stepsize: float = 0.08,
        interface: str = "autograd",
    ):
        if interface not in ("autograd", "jax"):
            raise ValueError(f"interface must be 'autograd' or 'jax', got {interface!r}")
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._num_wires = hamiltonian.num_wires
        self._device_name = device_name
        self._stepsize = stepsize
        self._interface = interface
        self._qml_h = hamiltonian.to_qml_hamiltonian()
        # Lazily built, then cached: (cost, grad_fn, adam_step). Built once
        # per solver instance so repeated run() calls -- e.g. every restart
        # inside run_multistart -- reuse the same jax.jit trace instead of
        # recompiling it from scratch on every call. See _get_jax_compiled.
        self._jax_compiled = None

    @property
    def num_wires(self) -> int:
        return self._num_wires

    @property
    def interface(self) -> str:
        return self._interface

    def _make_cost(self, interface: str):
        dev = qml.device(self._device_name, wires=self._num_wires)
        num_wires = self._num_wires
        ansatz = self._ansatz
        H = self._qml_h
        diff_method = "backprop"

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def cost(weights):
            ansatz.circuit(weights, num_wires)
            return qml.expval(H)

        return cost

    def run(self, seed: int = 0, steps: int = 300, verbose: bool = False) -> float:
        if self._interface == "jax":
            return self._run_jax(seed, steps, verbose)
        return self._run_autograd(seed, steps, verbose)

    def _run_autograd(self, seed: int, steps: int, verbose: bool) -> float:
        cost = self._make_cost("autograd")
        rng = np.random.default_rng(seed)
        shape = self._ansatz.param_shape(self._num_wires)
        weights = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=self._stepsize)
        for it in range(steps):
            weights, e = opt.step_and_cost(cost, weights)
            if verbose and it % 50 == 0:
                print(f"    step {it:4d}  E = {e:.6f}")
        return float(cost(weights))

    def _get_jax_compiled(self):
        """Builds (cost, grad_fn, adam_step) once and caches them on this
        instance. This matters: a jax.jit trace is keyed on the identity of
        the Python function object, so rebuilding ``jax.jit(...)`` from
        scratch inside every ``run()`` call -- the first version of this
        method did exactly that -- creates a fresh, cache-missing function
        every time and silently pays the full trace/compile cost on every
        restart of ``run_multistart``, defeating the entire point of JIT.
        Caching here means the shapes/dtypes only get traced once per
        solver instance, however many times ``run()`` is called after."""
        if self._jax_compiled is not None:
            return self._jax_compiled

        import jax
        import jax.numpy as jnp

        # JAX defaults to float32/complex64, silently truncating PennyLane's
        # float64/complex128 statevectors -- harmless for a 4-qubit demo,
        # but exactly the kind of silent-precision-loss gotcha that bites at
        # scale. Match PennyLane's default precision explicitly.
        jax.config.update("jax_enable_x64", True)

        cost = jax.jit(self._make_cost("jax"))
        grad_fn = jax.jit(jax.grad(cost))

        # Hand-rolled, jit-compiled Adam -- PennyLane's own AdamOptimizer
        # expects pennylane.numpy arrays, not jax arrays, so it can't be
        # reused here; this is the minimal state a JAX-native Adam needs.
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = self._stepsize

        @jax.jit
        def adam_step(weights, m, v, t, g):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)
            mhat = m / (1 - beta1 ** t)
            vhat = v / (1 - beta2 ** t)
            weights = weights - lr * mhat / (jnp.sqrt(vhat) + eps)
            return weights, m, v

        self._jax_compiled = (cost, grad_fn, adam_step)
        return self._jax_compiled

    def _run_jax(self, seed: int, steps: int, verbose: bool) -> float:
        try:
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover - exercised only without jax installed
            raise ImportError(
                "interface='jax' needs the optional jax/jaxlib dependencies: "
                'pip install -e ".[jax]"'
            ) from exc

        cost, grad_fn, adam_step = self._get_jax_compiled()

        rng = np.random.default_rng(seed)
        shape = self._ansatz.param_shape(self._num_wires)
        weights = jnp.array(rng.uniform(0, 2 * np.pi, size=shape))
        m = jnp.zeros_like(weights)
        v = jnp.zeros_like(weights)

        for it in range(1, steps + 1):
            g = grad_fn(weights)
            weights, m, v = adam_step(weights, m, v, it, g)
            if verbose and it % 50 == 0:
                print(f"    step {it:4d}  E = {float(cost(weights)):.6f}")

        return float(cost(weights))

    def run_multistart(self, n_restarts: int = 3, **kwargs) -> float:
        energies = [self.run(seed=seed, **kwargs) for seed in range(n_restarts)]
        return min(energies)
