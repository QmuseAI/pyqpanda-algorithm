import itertools

import numpy as np
import pytest
import sympy as sp

from pyqpanda_alg.BFDCQO.problems import IsingProblem
from pyqpanda_alg.BFDCQO.benchmark.upstream_qaoa import (
    UpstreamQAOASolver,
    ising_to_sympy,
    upstream_qaoa,
)
from pyqpanda_alg.QAOA import qaoa


def test_ising_to_sympy_preserves_every_basis_state_energy():
    model = IsingProblem(
        h=np.array([0.75, -1.25, 0.5]),
        J=np.array(
            [
                [0.0, 1.5, -0.25],
                [1.5, 0.0, 0.8],
                [-0.25, 0.8, 0.0],
            ]
        ),
        offset=-0.375,
    )

    expression = ising_to_sympy(model)
    symbols = sp.symbols("x0:3")

    for bits in itertools.product((0, 1), repeat=model.n):
        actual = float(expression.subs(dict(zip(symbols, bits))))
        assert actual == pytest.approx(model.energy_bits(bits), abs=1e-12)


def test_ising_to_sympy_rejects_an_unconstrained_variable():
    model = IsingProblem(
        h=np.array([1.0, 0.0]),
        J=np.zeros((2, 2)),
        offset=0.25,
    )

    with pytest.raises(ValueError, match="unconstrained"):
        ising_to_sympy(model)


def test_adapter_uses_repository_upstream_qaoa_class():
    assert upstream_qaoa is qaoa
    assert upstream_qaoa.QAOA is qaoa.QAOA
    assert upstream_qaoa.QAOA.__module__ == "pyqpanda_alg.QAOA.qaoa"


def test_real_cpuqvm_run_reports_observed_budget_and_little_endian_keys():
    # The unique optimum flips only q0.  In the public CPUQVM result format q0
    # is the right-most bit, so the non-palindromic optimum must be ``001``.
    model = IsingProblem(
        h=np.array([1.0, -1.0, -1.0]),
        J=np.zeros((3, 3)),
    )
    adapter = UpstreamQAOASolver(
        model,
        p=1,
        shots=128,
        max_evals=8,
        seed=0,
    )

    result = adapter.run()
    engine = adapter.qaoa_instance

    assert engine.__class__ is upstream_qaoa.QAOA
    assert result.distribution
    assert sum(result.distribution.values()) == pytest.approx(1.0, abs=1e-12)
    assert set(result.distribution) <= {format(i, "03b") for i in range(8)}
    assert max(result.distribution, key=result.distribution.get) == "001"
    assert model.energy_bitstring("001") < model.energy_bitstring("100")

    assert result.parameters.shape == (2,)
    assert np.all(np.isfinite(result.parameters))
    assert np.isfinite(result.loss)
    assert result.objective_evaluations == max(engine.iter - 1, 0)
    assert result.circuit_evaluations == engine.circuit_iter
    assert result.total_shots == result.circuit_evaluations * adapter.shots
    assert result.total_shots > adapter.shots
