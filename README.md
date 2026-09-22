# cluster-mbqc-toolkit

A small PennyLane toolkit for **rotated 1D cluster states**, built to
extend my M.Sc. thesis — *"Conditional Information Flow and Neural
Representability of One-Dimensional Rotated Cluster States"* (Perimeter
Institute / University of Waterloo) — into working quantum-circuit code.

The thesis studies rotated cluster states with classical tools: neural
quantum states (RBM, autoregressive RNN), MPS/DMRG, and the conditional
mutual information (CMI) of their measurement outcomes. This repo asks a
different question: **what does the same physics look like natively as
PennyLane circuits**, and what does that require in terms of software
design, not just one-off scripts?

Every numerical claim below is checked in `tests/` (50 tests, all passing)
or in the runnable `examples/`, cross-checked at least two independent
ways where the physics allows it (e.g. against a hand-written brute-force
`numpy` matrix conjugation with no PennyLane involved) — nothing here is
asserted without a test backing it.

## Physics results this toolkit reproduces

1. **Conditional mutual information** of the rotated cluster/entanglement-
   swapped state's measurement distribution matches the thesis's exact
   endpoint result: `I(X_1:X_N | X_2..X_{N-1})` is exactly 1 bit at
   `theta=0` and exactly 0 at `theta=pi/2`, for N up to 12 — computed via a
   real PennyLane circuit's `qml.probs()`, not the thesis's own hand-written
   MPS contraction.
2. A `default.tensor` (quimb-backed MPS, bond dimension 2) simulation
   reproduces the exact statevector to fidelity 1, confirming from the
   circuit side that cluster states are *exactly* bond-dimension-2 MPS.
3. A **single-qubit MBQC gate-teleportation gadget** (`qml.measure()` +
   `qml.cond()`) reproduces a direct `H . PhaseShift(-theta)` gate to
   fidelity 1 across random input states and angles — the thing the
   thesis's MBQC section and its own `mbqc.ipynb` notebook both point at
   but neither builds.
4. **VQE** finds the exact ground energy `E0 = -N` of the cluster
   Hamiltonian `H_C`.
5. Perhaps the least obvious result: the ground energy of the *rotated*
   Hamiltonian `H(theta) = U(theta) H_C U(theta)^dagger` is **exactly `-N`
   for every theta**, not just `theta=0` — because conjugating by a purely
   local unitary can never change a Hamiltonian's spectrum. Energy is
   blind to exactly the structure CMI is built to detect; that contrast is
   the actual point of both this repo and the thesis chapter it extends.

## Design

The physics is simple enough that a handful of top-level functions would
"work" — the point of structuring it as a package instead was to practice
the kind of interfaces a shared quantum-software codebase actually needs:
something that stays correct when a second person adds a new circuit,
Hamiltonian, or ansatz without having to read (or risk breaking) the rest
of the file.

```
StatePreparationCircuit (ABC)              StabilizerHamiltonian (ABC)
├── ClusterStateCircuit                     ├── ClusterHamiltonian
│   └── EntanglementSwappedCircuit          RotatedHamiltonian  (wraps any
RotatedCircuit  (wraps any                      StabilizerHamiltonian)
    StatePreparationCircuit)

Ansatz (ABC)                                CMIAnalyzer   (encapsulation)
└── HardwareEfficientAnsatz                 MBQCGadget    (encapsulation)

VQESolver(hamiltonian: StabilizerHamiltonian, ansatz: Ansatz, ...)
```

A few choices worth spelling out, since they're the actual content of
"software design" here rather than decoration:

- **Interface + template method** (`StatePreparationCircuit`,
  `StabilizerHamiltonian`): a subclass implements exactly one method
  (`build()` / `terms()`); the base class turns that into working
  `probs()`/`state()` or `to_qml_hamiltonian()`/`matrix()`/
  `ground_energy_exact()` methods for free. Adding a new circuit or
  Hamiltonian never means re-deriving the surrounding plumbing.
- **Inheritance where the relationship is IS-A**:
  `EntanglementSwappedCircuit` subclasses `ClusterStateCircuit` and extends
  `build()` with `super().build()`, because the ES protocol genuinely *is*
  a cluster-state preparation plus one more step.
- **Composition where the added behaviour is orthogonal to the type**:
  `RotatedCircuit` and `RotatedHamiltonian` *wrap* a circuit/Hamiltonian
  instance instead of subclassing a concrete type, because "apply a
  per-site rotation" should work identically on a cluster state, an ES
  state, or anything added later — multiplying that behaviour into every
  subclass (or via multiple inheritance) would be the wrong tool here.
- **Dependency injection / strategy pattern** (`VQESolver`): takes a
  `StabilizerHamiltonian` and an `Ansatz` as constructor arguments rather
  than hard-coding either. Swapping `ClusterHamiltonian` for
  `RotatedHamiltonian(ClusterHamiltonian(N), theta)`, or the
  hardware-efficient ansatz for a physics-informed one, never touches
  `VQESolver` itself — see `tests/test_vqe.py::test_solver_is_ansatz_agnostic`
  for a from-scratch minimal `Ansatz` plugged in with zero solver changes.
- **Encapsulation with the validation living on the object**
  (`CMIAnalyzer`, `MBQCGadget`): `MBQCGadget.verify()` is a method on the
  same class as `MBQCGadget.apply()`, so "does this actually do what it
  claims" travels with the implementation instead of living in a
  disconnected, easy-to-forget test script.

## Install

```bash
pip install -e ".[dev]"
# optional, only needed for the default.tensor (MPS) demo in example 1:
pip install -e ".[tensor]"
```

## Run

```bash
pytest                              # 50 tests, ~30s
python examples/01_cmi_crosscheck.py
python examples/02_mbqc_gadget.py
python examples/03_vqe_ground_state.py   # a few minutes (N up to 8)
python examples/04_vqe_theta_sweep.py    # ~10 minutes (see note below)
```

## A genuine trainability finding (not swept under the rug)

`examples/03` and `04` use the same hardware-efficient ansatz (rounds of
RY/RZ + a ring of CNOTs). At `theta=0` a 6-layer version converges to the
exact energy to ~1e-8. Away from `theta=0`, the *same* 6-8 layer ansatz
reproducibly stalled 0.5-8% per site short, across 8 independent random
restarts — not bad luck, a real harder optimization landscape once
`H(theta)`'s Pauli terms stop lining up with the ansatz's native structure.
Going to 10-12 layers with a smaller Adam step size (0.05-0.06 instead of
0.08) fixed it. The exact answer was never in doubt — exact diagonalization
independently proves `E0(theta) = -N` — this was purely about how hard the
*optimization* is, which is precisely the distinction the thesis's own
comparison of classical methods (RBM / autoregressive RNN / MPS-induced
recurrent model) draws for a different reason: representability and
trainability are different axes of difficulty, and a generic ansatz pays
for not having the problem's structure (bond-dimension-2 / stabilizer
commutativity) baked in.

## Lessons learned getting `quimb` running (kept here on purpose)

While testing `default.tensor` locally, `pip install quimb` alone broke an
existing PennyLane install with `AttributeError: module 'autoray.autoray'
has no attribute 'NumpyMimic'`. Root cause: PennyLane pins
`autoray<0.8`, but quimb's `cotengra` dependency needs
`autoray>=0.8`'s `get_namespace()` — the two most recent published
versions of these packages are mutually incompatible as of this writing.
Fix: pin both together to a version where `cotengra` doesn't need the
newer API:

```bash
pip install "autoray==0.7.2" "cotengra==0.7.5"
```

`cluster_mbqc_toolkit.devices.make_device()` catches exactly this failure
mode and re-raises with the fix inline (see `tests/test_devices.py`),
instead of leaving the next person to rediscover it from a generic
"pip install quimb" message that doesn't actually solve it.

## What this repo is (and isn't)

This is personal project work, built to get hands-on with PennyLane while
extending my own M.Sc. research — it isn't affiliated with Xanadu. The
thesis PDF and full research codebase (neural-network models, MPS/DMRG
baselines, NetKet) live in a separate repository; this one is scoped
specifically to the parts of that work that map onto PennyLane circuits.
