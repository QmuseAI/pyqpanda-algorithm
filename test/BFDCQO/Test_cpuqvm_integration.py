import math

import numpy as np
from pyqpanda3.core import CPUQVM, QProg, RY, measure

from pyqpanda_alg.BFDCQO.backends.numpy_statevector import (
    apply_pauli_rotation,
    probabilities,
    product_state_from_ry,
)
from pyqpanda_alg.BFDCQO.backends.pyqpanda3_backend import (
    pauli_rotation_recipe,
    run_bfdcqo_probabilities,
)


def test_mixed_pauli_rotation_recipe_has_complete_gate_order():
    recipe = pauli_rotation_recipe(("X", "Y", "I", "Z"), 0.37)

    assert recipe == [
        ("H", 0),
        ("RX", 1, math.pi / 2.0),
        ("CNOT", 0, 3),
        ("CNOT", 1, 3),
        ("RZ", 3, 0.74),
        ("CNOT", 1, 3),
        ("CNOT", 0, 3),
        ("RX", 1, -math.pi / 2.0),
        ("H", 0),
    ]


def test_cpuqvm_mixed_pauli_rotation_matches_numpy_probabilities():
    angles = np.array([0.31, -0.42, 0.18, 0.77])
    pauli_string = ("X", "Y", "I", "Z")
    theta = 0.37
    expected = probabilities(
        apply_pauli_rotation(
            product_state_from_ry(angles),
            pauli_string,
            theta,
        )
    )

    actual = run_bfdcqo_probabilities(
        n=4,
        ry_angles=angles,
        trotter_layers=[[(pauli_string, theta)]],
    )

    assert np.allclose(actual, expected, atol=1e-8, rtol=0.0)


def test_cpuqvm_probability_indices_are_little_endian():
    actual = run_bfdcqo_probabilities(
        n=3,
        ry_angles=np.array([math.pi, 0.0, math.pi]),
        trotter_layers=[],
    )

    assert int(np.argmax(actual)) == 0b101


def test_cpuqvm_count_bitstrings_match_probability_index_order():
    shots = 128
    program = QProg(3)
    for qubit, angle in enumerate((math.pi, 0.0, math.pi)):
        program << RY(qubit, angle)
    program << measure(list(range(3)), list(range(3)))

    machine = CPUQVM()
    machine.run(program, shots)
    counts = {
        str(bits).replace(" ", ""): int(count)
        for bits, count in machine.result().get_counts().items()
    }

    assert counts == {"101": shots}
    assert sum(counts.values()) == shots
