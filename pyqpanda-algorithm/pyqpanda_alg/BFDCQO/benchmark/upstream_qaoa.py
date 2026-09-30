# Licensed under the Apache License, Version 2.0.
"""Adapter for the repository's public PyQPanda QAOA implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import sympy as sp

from pyqpanda_alg.QAOA import qaoa as upstream_qaoa

from ..problems import IsingProblem


def ising_to_sympy(model: IsingProblem) -> sp.Expr:
    """Convert ``model`` to the binary expression expected by upstream QAOA.

    The conversion substitutes :math:`z_i = 1 - 2x_i` directly into the
    quadratic Ising objective.  The upstream implementation infers its qubit
    count from free symbols, so a model containing an unconstrained variable
    is rejected instead of silently running a lower-dimensional problem.
    """

    if not isinstance(model, IsingProblem):
        raise TypeError("model must be an IsingProblem")

    variables = sp.symbols(f"x0:{model.n}")
    spins = [1 - 2 * variable for variable in variables]
    expression = sp.Float(model.offset)

    for i in range(model.n):
        expression += sp.Float(model.h[i]) * spins[i]
        for j in range(i + 1, model.n):
            expression += sp.Float(model.J[i, j]) * spins[i] * spins[j]

    expression = sp.expand(expression)
    if expression.free_symbols != set(variables):
        missing = sorted(
            set(variables) - expression.free_symbols,
            key=lambda symbol: symbol.name,
        )
        missing_names = ", ".join(str(symbol) for symbol in missing)
        raise ValueError(
            "upstream QAOA cannot represent unconstrained variables; "
            f"missing from the simplified objective: {missing_names}"
        )
    return expression


@dataclass(frozen=True)
class UpstreamQAOAResult:
    """Observed output and execution budget of one upstream QAOA run."""

    distribution: dict[str, float]
    parameters: np.ndarray
    loss: float
    objective_evaluations: int
    circuit_evaluations: int
    shots_per_circuit: int
    total_shots: int
    backend: str = "pyqpanda3.CPUQVM"
    implementation: str = "pyqpanda_alg.QAOA.qaoa.QAOA"


class UpstreamQAOASolver:
    """Run the repository's public QAOA solver on its real CPUQVM path.

    Upstream CPUQVM distributions already use the convention that q0 is the
    right-most (least-significant) bit.  Keys are therefore validated and
    preserved unchanged to match :meth:`IsingProblem.energy_bitstring`.
    """

    def __init__(
        self,
        model: IsingProblem,
        *,
        p: int = 1,
        shots: int = 1024,
        max_evals: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(model, IsingProblem):
            raise TypeError("model must be an IsingProblem")
        if not isinstance(p, int) or p <= 0:
            raise ValueError("p must be a positive integer")
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if not isinstance(max_evals, int) or max_evals <= 0:
            raise ValueError("max_evals must be a positive integer")

        self.model = model
        self.p = p
        self.shots = shots
        self.max_evals = max_evals
        self.seed = seed
        self._qaoa_instance: Optional[upstream_qaoa.QAOA] = None

    @property
    def qaoa_instance(self) -> upstream_qaoa.QAOA:
        """Return the concrete upstream instance after :meth:`run`."""

        if self._qaoa_instance is None:
            raise RuntimeError("QAOA has not been run yet")
        return self._qaoa_instance

    def run(self) -> UpstreamQAOAResult:
        """Optimize and sample using upstream QAOA and a PyQPanda3 CPUQVM."""

        expression = ising_to_sympy(self.model)
        engine = upstream_qaoa.QAOA(expression)
        self._qaoa_instance = engine

        random_state = np.random.get_state()
        try:
            if self.seed is not None:
                np.random.seed(self.seed)
            distribution, parameters, loss = engine.run(
                layer=self.p,
                shots=self.shots,
                optimizer="COBYLA",
                optimizer_option={"options": {"maxiter": self.max_evals}},
            )
        finally:
            np.random.set_state(random_state)

        normalized_distribution = {
            str(bitstring): float(probability)
            for bitstring, probability in distribution.items()
        }
        invalid_keys = [
            bitstring
            for bitstring in normalized_distribution
            if len(bitstring) != self.model.n
            or any(bit not in "01" for bit in bitstring)
        ]
        if invalid_keys:
            raise RuntimeError(
                "upstream QAOA returned invalid computational-basis keys: "
                f"{invalid_keys!r}"
            )

        objective_evaluations = max(int(engine.iter) - 1, 0)
        circuit_evaluations = int(engine.circuit_iter)
        return UpstreamQAOAResult(
            distribution=normalized_distribution,
            parameters=np.asarray(parameters, dtype=float).copy(),
            loss=float(loss),
            objective_evaluations=objective_evaluations,
            circuit_evaluations=circuit_evaluations,
            shots_per_circuit=self.shots,
            total_shots=circuit_evaluations * self.shots,
        )
