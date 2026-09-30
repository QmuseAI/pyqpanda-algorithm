# PyBF-DCQO for PyQPanda

`BFDCQO` adds a modular family of bias-field digitized counterdiabatic quantum optimization solvers to `pyqpanda-algorithm`.

The implementation is designed for two purposes:

1. provide a transparent PyQPanda3 implementation of the published BF-DCQO protocol and its higher-order/CVaR extensions; and
2. provide clearly separated framework extensions for finite-shot allocation and bounded hybrid refinement, with reproducible small-instance benchmarks.

## Features

| Capability | API | Notes |
|---|---|---|
| Basic BF-DCQO | `BFDCQOConfig(variant="basic")` | Published BF-DCQO feedback protocol |
| CVaR feedback | `variant="cvar"` | Low-energy sample filtering; literature-derived enhancement |
| Adaptive shots | `variant="adaptive"` | Fixed total-shot budget redistributed across feedback rounds |
| Adaptive + CVaR | `variant="adaptive_cvar"` | Combines finite-shot allocation and filtered feedback |
| Hybrid refinement | `variant="hybrid"` | Bounded confidence-guided or random-block local refinement |
| Native QUBO | `IsingProblem` / `MaxCutProblem` / `MWISProblem` | Quadratic Ising objectives |
| Native HUBO | `PolynomialIsingProblem` | Arbitrary-order diagonal Pauli-Z objectives |
| MAX-3-SAT | `Max3SATProblem` | Native cubic HUBO plus optional Rosenberg QUBO reduction |
| Exact reference | `ExactSolver` | Ground truth for small regression instances |
| QAOA benchmark | `QAOASolver` | Transparent p-layer QAOA reference adapter |

## Unified API

```python
from pyqpanda_alg.BFDCQO import BFDCQOConfig, BFDCQOSolver, MaxCutProblem

problem = MaxCutProblem.from_edges(
    5,
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
).to_ising()

config = BFDCQOConfig(
    variant="adaptive_cvar",
    iterations=6,
    shots=512,
    total_shot_budget=3072,
    cvar_alpha=0.10,
)

result = BFDCQOSolver(
    problem,
    config,
    backend="pyqpanda",
    seed=7,
).solve()

print(result.best_solution, result.best_energy, result.total_shots)
```

For deterministic reference validation, use `backend="numpy"`. The recommended production-facing entry point is the unified `BFDCQOSolver`; low-level solver classes remain public for research experiments.

## Native HUBO

A diagonal higher-order objective is represented directly as

$$
H = c + \sum_S J_S \prod_{i\in S} Z_i.
$$

For example:

```python
from pyqpanda_alg.BFDCQO import PolynomialIsingProblem

problem = PolynomialIsingProblem(
    n=4,
    terms={
        (0,): 0.3,
        (0, 1): -0.8,
        (0, 1, 3): 1.2,
    },
    offset=0.1,
)
```

The first-order nested-commutator adiabatic gauge potential is constructed with sparse Pauli algebra, so the same implementation supports two-body and higher-order terms. The PyQPanda3 backend decomposes arbitrary Pauli rotations through basis changes and a parity CNOT ladder.

## Solver variants

### Basic BF-DCQO

Each round prepares the biased product-state ground state, applies digitized counterdiabatic evolution, measures the output distribution, and uses the resulting $\langle Z_i\rangle$ values as the next longitudinal bias.

### CVaR BF-DCQO

Only the lowest-energy fraction `cvar_alpha` of sampled probability mass is used to update the bias. The implementation treats a cutoff crossing a degenerate energy level proportionally, avoiding an artificial bitstring-order bias.

### Adaptive-shot BF-DCQO

A total sampling budget is reserved up front. Later rounds receive more precision as the mean bias confidence increases, while a minimum allocation is reserved for every remaining round. Optional early stopping uses both uncertainty and bias change.

### Hybrid BF-DCQO

The final all-shot bias estimates define variable confidence $c_i=|\langle Z_i\rangle|$. A small exact block neighborhood is then searched over the least-confident variables. The local evaluation count is explicit in the result and can be matched against a random-block ablation.

## Applications and benchmarks

Included examples cover:

- weighted MaxCut (QUBO),
- weighted maximum independent set (MWIS, QUBO),
- weighted MAX-3-SAT (native cubic HUBO), and
- native-HUBO versus quadratized-QAOA resource/quality comparisons.

The competition benchmark artifacts live outside the source package under `benchmark/`. They deliberately separate **algorithm code** from **reproducibility evidence**.

## Correctness strategy

The test suite checks, among other items:

- QUBO/HUBO energy preservation for every basis state of small fixtures;
- sparse-Pauli commutators against dense-matrix calculations;
- Pauli rotations against `scipy.linalg.expm`;
- biased product-state preparation;
- shot-budget conservation;
- CVaR degeneracy handling;
- native-HUBO MAX-3-SAT encoding and Rosenberg quadratization;
- bounded confidence-guided versus random local-search budgets; and
- NumPy/PyQPanda3 distribution equality when PyQPanda3 is available.

## Scope and limitations

This contribution is currently a **small-instance algorithm/reference implementation and benchmark suite**. Exact ground-state diagnostics and the NumPy reference backend enumerate the full computational basis. Before using the same code path for large hardware instances, exact diagnostics should be made optional and hardware execution should consume sampled counts without materializing a full statevector.

Benchmark numbers bundled with the competition package were generated with the independent NumPy statevector reference backend because PyQPanda3 was unavailable in the build container. They are not labelled as CPUQVM results. A no-fallback CPUQVM cross-check script is supplied for local validation.

## References and attribution

The implementation is independent and based on public equations; no proprietary Kipu Quantum implementation code is included.

- A. Gomez Cadavid *et al.*, **Bias-field digitized counterdiabatic quantum optimization**, Phys. Rev. Research 7, L022010 (2025), arXiv:2405.13898.
- S. V. Romero *et al.*, **Bias-Field Digitized Counterdiabatic Quantum Algorithm for Higher-Order Binary Optimization**, Communications Physics 8, 348 (2025), arXiv:2409.04477.
- P. Chandarana *et al.*, **Hybrid Sequential Quantum Computing**, arXiv:2510.05851 (2025).

See `INNOVATION_BOUNDARY.md` in the competition package for an explicit prior-work/contribution matrix.
