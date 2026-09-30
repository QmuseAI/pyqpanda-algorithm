# Licensed under the Apache License, Version 2.0.
"""End-to-end checks for the reproducible CPUQVM benchmark driver."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_bfdcqo_cpuqvm_benchmarks.py"


def test_quick_cli_writes_real_cpuqvm_evidence(tmp_path):
    env = os.environ.copy()
    package_root = str(ROOT / "pyqpanda-algorithm")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (package_root, env.get("PYTHONPATH", "")) if part
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--output", str(tmp_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary_names = (
        "maxcut_summary.csv",
        "max3sat_summary.csv",
        "family_summary.csv",
        "mwis_summary.csv",
    )
    for name in summary_names:
        path = tmp_path / name
        assert path.is_file(), name
        with path.open(newline="", encoding="utf-8") as handle:
            assert next(csv.DictReader(handle), None) is not None

    metadata = json.loads(
        (tmp_path / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["mode"] == "quick"
    assert metadata["seeds"] == [0]
    assert metadata["backends"] == {
        "bfdcqo": "pyqpanda3.CPUQVM",
        "qaoa": "pyqpanda_alg.QAOA.CPUQVM",
    }
    assert metadata["qaoa_budget_accounting"] == (
        "observed circuit_evaluations * shots"
    )
    assert metadata["sampling_seed_controlled"] is False

    for raw_name in (
        "maxcut_raw.csv",
        "max3sat_raw.csv",
        "family_raw.csv",
        "mwis_raw.csv",
    ):
        assert (tmp_path / raw_name).is_file()
