# cluster-mbqc-toolkit

[![tests](https://github.com/HelioXv022/cluster-mbqc-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/HelioXv022/cluster-mbqc-toolkit/actions/workflows/test.yml)

A PennyLane package for **rotated 1D cluster states**. It implements the
physics of my M.Sc. thesis, *"Conditional Information Flow and Neural
Representability of One-Dimensional Rotated Cluster States"* (Perimeter
Institute / University of Waterloo), as quantum circuits.

The thesis studies rotated cluster states with classical methods: neural
quantum states (RBM, autoregressive RNN), MPS/DMRG, and the conditional
mutual information (CMI) of measurement outcomes. This package builds the
same states, Hamiltonians and measurements as PennyLane circuits, organized
as a tested, extensible library.

Every numerical result below is verified by the test suite (52 tests, run
by CI on every push) or by a runnable script in `examples/`. Where
possible, results are checked two independent ways, e.g. against a
brute-force `numpy` matrix conjugation that does not use PennyLane.

## Results

1. **Conditional mutual information.** The endpoint CMI
   `I(X_1:X_N | X_2..X_{N-1})` of the rotated cluster / entanglement-swapped
   state is exactly 1 bit at `theta=0` and exactly 0 at `theta=pi/2`, for N
   up to 12, computed from `qml.probs()` of a PennyLane circuit. This
   matches the thesis's analytic result.
2. **Exact MPS representation.** A `default.tensor` simulation (quimb MPS,
   bond dimension 2) reproduces the exact statevector with fidelity 1:
   cluster states are exactly bond-dimension-2 MPS.
3. **MBQC gate teleportation.** A single-qubit gadget built from
   `qml.measure()` and `qml.cond()` implements `H · PhaseShift(-theta)` with
   fidelity 1 over random input states and angles. The thesis describes
   this gadget; this package implements and verifies it.
4. **VQE ground state.** VQE reaches the exact ground energy `E0 = -N` of
   the cluster Hamiltonian `H_C`.
5. **Rotation changes the CMI but not the energy.** The rotated
   Hamiltonian `H(theta) = U(theta) H_C U(theta)^dagger` has ground energy
   exactly `-N` for every `theta`, because conjugation by a local unitary
   preserves the spectrum. The CMI changes with `theta` (result 1). Energy
   cannot detect the structure that CMI measures; this contrast is the
   central point of the thesis chapter this package extends.

## Design

A new circuit, Hamiltonian, or ansatz can be added without modifying any
existing class.

```
StatePreparationCircuit (ABC)              StabilizerHamiltonian (ABC)
├── ClusterStateCircuit                     ├── ClusterHamiltonian
│   └── EntanglementSwappedCircuit          RotatedHamiltonian  (wraps any
RotatedCircuit  (wraps any                      StabilizerHamiltonian)
    StatePreparationCircuit)

Ansatz (ABC)                                CMIAnalyzer   (encapsulation)
└── HardwareEfficientAnsatz                 MBQCGadget    (encapsulation)

VQESolver(hamiltonian: StabilizerHamiltonian, ansatz: Ansatz,
          interface: "autograd" | "jax", ...)
```

- **Interface + template method** (`StatePreparationCircuit`,
  `StabilizerHamiltonian`): a subclass implements one method (`build()` /
  `terms()`); the base class provides `probs()` / `state()` or
  `to_qml_hamiltonian()` / `matrix()` / `ground_energy_exact()` on top of
  it.
- **Inheritance for is-a relationships**: `EntanglementSwappedCircuit`
  subclasses `ClusterStateCircuit` and extends `build()` via
  `super().build()`. The entanglement-swapping protocol is a cluster-state
  preparation plus one additional step.
- **Composition for orthogonal behavior**: `RotatedCircuit` and
  `RotatedHamiltonian` wrap any circuit or Hamiltonian instance. A per-site
  rotation is independent of which state it is applied to, so it is
  implemented once as a wrapper, not repeated in every subclass or mixed in
  through multiple inheritance.
- **Dependency injection** (`VQESolver`): the Hamiltonian, the ansatz, and
  the autodiff backend are constructor arguments. Replacing
  `ClusterHamiltonian` with `RotatedHamiltonian(ClusterHamiltonian(N),
  theta)`, swapping the ansatz, or switching `interface="autograd"` to
  `interface="jax"` requires no change to `VQESolver`.
  `tests/test_vqe.py::test_solver_is_ansatz_agnostic` plugs in a new
  `Ansatz`; `test_jax_interface_matches_autograd_interface` checks that
  both backends reach the same energy.
- **Encapsulation with self-verification** (`CMIAnalyzer`, `MBQCGadget`):
  `MBQCGadget.verify()` lives on the same class as `MBQCGadget.apply()`, so
  the correctness check ships with the implementation.

## Install

```bash
pip install -e ".[dev]"
pip install -e ".[tensor]"   # default.tensor (MPS) demo in example 1
pip install -e ".[jax]"      # VQESolver(..., interface="jax") and example 5
```

## Run

```bash
pytest                                        # 52 tests, ~45 s
python examples/01_cmi_crosscheck.py
python examples/02_mbqc_gadget.py
python examples/03_vqe_ground_state.py        # a few minutes (N up to 8)
python examples/04_vqe_theta_sweep.py         # ~10 minutes
python examples/05_jax_vs_autograd_speed.py   # ~30 s, needs the jax extra
```

## Finding: VQE trainability away from theta = 0

Examples 03 and 04 use a hardware-efficient ansatz (RY/RZ layers plus a
ring of CNOTs). At `theta=0`, 6 layers reach the exact energy to ~1e-8.
Away from `theta=0`, the same 6–8-layer ansatz stops 0.5–8% per site above
the exact energy, consistently across 8 random restarts. 10–12 layers with
an Adam step size of 0.05–0.06 (instead of 0.08) reach the exact energy.

Exact diagonalization gives `E0(theta) = -N` for every `theta`, so the gap
is an optimization effect, not an error in the Hamiltonian. This mirrors
the thesis's comparison of RBM, autoregressive RNN and MPS-induced models:
representability and trainability are separate difficulties, and an ansatz
that does not encode the problem's structure (bond dimension 2, commuting
stabilizers) is harder to train.

## Finding: JIT compilation cost and caching

`VQESolver(..., interface="jax")` runs the same optimization with a
`jax.jit`-compiled cost, gradient, and Adam update
(`examples/05_jax_vs_autograd_speed.py`). Measured at N=6, 8-layer ansatz,
80 Adam steps per run:

| backend  | first `run()`              | later `run()` calls, same solver |
|----------|----------------------------|----------------------------------|
| autograd | ~8 s                       | ~8 s                             |
| jax      | ~35 s (trace + compile)    | ~0.04 s                          |

JAX is slower for a single run and ~200× faster for every run after the
first. Use `interface="jax"` for `run_multistart`, `theta` sweeps, and any
workload that calls `run()` repeatedly on the same solver.

The speed-up depends on `VQESolver` caching the compiled functions on the
instance (`_get_jax_compiled()`). The first implementation rebuilt
`jax.jit(...)` inside every `run()` call; each call created a new function
object, JAX's compilation cache missed, and every restart in
`run_multistart` paid the full compile cost. The energies were correct, so
the unit tests passed; timing the two backends side by side exposed the
bug. Caching the compiled functions on the instance fixed it.

## Known issue: jax ≥ 0.11 (caught by CI)

jax 0.11.0 (July 2026) removed `jax.core.is_concrete`. PennyLane 0.45.1
still calls it in `pennylane/math/utils.py::is_abstract`, so
`interface="jax"` fails with:

```
AttributeError: jax.core.is_concrete was deprecated in JAX v0.10.0 and removed in JAX v0.11.0.
```

Local tests passed because the development environment had jax 0.10.2,
where the call only raises a `DeprecationWarning`. The first CI run
installed jax 0.11.2 and failed. Fix: the `jax` extra pins `jax<0.11` and
`jaxlib<0.11`.

## Known issue: PennyLane ≤ 0.42 and quimb

PennyLane ≤ 0.42 pins `autoray<0.8`. Current quimb installs a cotengra
that requires `autoray>=0.8`, and autoray 0.8 removed `NumpyMimic`, which
PennyLane ≤ 0.42 uses. Installing quimb on PennyLane ≤ 0.42 therefore
breaks `import pennylane` with `AttributeError: module 'autoray.autoray'
has no attribute 'NumpyMimic'`.

Fixes:

- Use PennyLane ≥ 0.43, which pins `autoray==0.8.x` (verified with
  PennyLane 0.45.1, quimb 1.15.0, cotengra 0.8.2). The `tensor` extra
  requires `pennylane>=0.43`.
- To stay on PennyLane ≤ 0.42, pin both packages:
  `pip install "autoray==0.7.2" "cotengra==0.7.5"`.

When `default.tensor` cannot load quimb, `make_device()` raises an
`ImportError` that states both fixes (tested in `tests/test_devices.py`).

## Roadmap

- **Compile the full multistart loop with Catalyst.**
  [Catalyst](https://github.com/PennyLaneAI/catalyst)'s `@qjit` compiles an
  entire hybrid program, including the classical optimization loop,
  through an MLIR quantum dialect to LLVM/QIR. `VQESolver` currently
  JIT-compiles one QNode call at a time; compiling all of
  `run_multistart` with `@qjit` extends the same compile-once,
  reuse-many-times approach to the whole workload.

## Scope

Personal project, not affiliated with Xanadu. It covers the parts of my
M.Sc. research that map onto quantum circuits. The thesis and the full
research codebase (neural quantum state models, MPS/DMRG baselines,
NetKet) are in a separate repository.
