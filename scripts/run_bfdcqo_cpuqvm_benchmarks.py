# Licensed under the Apache License, Version 2.0.
"""Run the reproducible BF-DCQO CPUQVM evidence suite.

Raw run data is written only below the caller-provided output directory.  The
optional documentation output contains aggregated tables, two figures, and a
Chinese report; it deliberately excludes raw per-run records.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Iterable, Mapping

from pyqpanda_alg.BFDCQO.benchmark import (
    default_max3sat_instances,
    default_mwis_instances,
    run_family_benchmark,
    run_max3sat_benchmark,
    run_maxcut_benchmark,
    run_mwis_hybrid_benchmark,
)
from pyqpanda_alg.BFDCQO.benchmark.instances import default_maxcut_instances


QUICK = {
    "seeds": (0,),
    "family_budgets": (1536,),
    "maxcut_qaoa_shots": 256,
    "maxcut_qaoa_max_evals": 22,
    "mwis_budget": 72,
    "max3sat_instance_count": 1,
    "mwis_instance_count": 1,
}

OFFICIAL = {
    "seeds": (0, 1, 2),
    "family_budgets": (1536, 3072, 6144),
    "maxcut_qaoa_shots": 256,
    "maxcut_qaoa_max_evals": 22,
    "mwis_budget": 72,
    # Exact verification and statevector reference costs remain tractable.
    "max3sat_instance_count": 2,
    "mwis_instance_count": 3,
}

BF_BACKEND = "pyqpanda3.CPUQVM"
QAOA_BACKEND = "pyqpanda_alg.QAOA.CPUQVM"


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty benchmark table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in fieldnames})


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    return sum(materialized) / len(materialized)


def _population_std(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    average = _mean(materialized)
    return (_mean((value - average) ** 2 for value in materialized)) ** 0.5


def _matching_rows(
    rows: list[dict[str, object]], summary: Mapping[str, object], keys: tuple[str, ...]
) -> list[dict[str, object]]:
    return [
        row for row in rows
        if all(row[key] == summary[key] for key in keys)
    ]


def _enrich_summary(
    payload: dict[str, object], *, keys: tuple[str, ...], metric: str,
    target_key: str = "target_shot_budget",
) -> list[dict[str, object]]:
    rows = payload["rows"]
    result = []
    for original in payload["summary"]:
        selected = _matching_rows(rows, original, keys)
        target_values = [float(row[target_key]) for row in selected]
        observed_values = [float(row["total_shots"]) for row in selected]
        deltas = [
            float(row.get("budget_delta", observed - target))
            for row, observed, target in zip(
                selected, observed_values, target_values
            )
        ]
        result.append({
            **original,
            "backend": selected[0]["backend"],
            "seed_count": len({int(row["seed"]) for row in selected}),
            f"std_{metric}": _population_std(float(row[metric]) for row in selected),
            "mean_target_shot_budget": _mean(target_values),
            "mean_observed_total_shots": _mean(observed_values),
            "mean_budget_delta": _mean(deltas),
        })
    return result


def _family_summary(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = payload["rows"]
    result = []
    for original in payload["summary"]:
        selected = _matching_rows(
            rows, original, ("problem", "variant", "shot_budget")
        )
        result.append({
            **original,
            "backend": selected[0]["backend"],
            "seed_count": len({int(row["seed"]) for row in selected}),
            "mean_target_shot_budget": float(original["shot_budget"]),
            "mean_observed_total_shots": _mean(
                float(row["total_shots"]) for row in selected
            ),
            "mean_budget_delta": _mean(
                float(row["total_shots"]) - float(row["shot_budget"])
                for row in selected
            ),
        })
    return result


def _metadata(repository: Path, config: Mapping[str, object], quick: bool) -> dict[str, object]:
    return {
        "mode": "quick" if quick else "official",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pyqpanda3_version": importlib.metadata.version("pyqpanda3"),
        "package_version": "2.0.0",
        "git_commit": _git_commit(repository),
        "seeds": list(config["seeds"]),
        "configuration": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in config.items()
        },
        "instance_counts": {
            "maxcut": len(default_maxcut_instances()),
            "max3sat": int(config["max3sat_instance_count"]),
            "family": 2,
            "mwis": int(config["mwis_instance_count"]),
        },
        "backends": {"bfdcqo": BF_BACKEND, "qaoa": QAOA_BACKEND},
        "initial_parameter_seed_scope": (
            "initial_parameter_seed controls optimizer initialization only"
        ),
        "sampling_seed_controlled": False,
        "sampling_note": (
            "CPUQVM shot sampling is not controlled by initial_parameter_seed"
        ),
        "qaoa_budget_accounting": "observed circuit_evaluations * shots",
    }


def run_suite(output_dir: Path, *, quick: bool) -> dict[str, object]:
    """Run quick or official CPUQVM evidence and write raw + aggregate files."""
    repository = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = QUICK if quick else OFFICIAL
    seeds = config["seeds"]

    maxcut = run_maxcut_benchmark(
        seeds=seeds,
        bfdcqo_backend="pyqpanda",
        qaoa_implementation="upstream",
        qaoa_shots=config["maxcut_qaoa_shots"],
        qaoa_max_evals=config["maxcut_qaoa_max_evals"],
    )
    max3sat_instances = tuple(default_max3sat_instances())[
        : config["max3sat_instance_count"]
    ]
    max3sat = run_max3sat_benchmark(
        seeds=seeds,
        instances=max3sat_instances,
        bfdcqo_backend="pyqpanda",
        qaoa_implementation="upstream",
        qaoa_shots=config["maxcut_qaoa_shots"],
        qaoa_max_evals=config["maxcut_qaoa_max_evals"],
    )
    family = run_family_benchmark(
        budgets=config["family_budgets"],
        seeds=seeds,
        backend="pyqpanda",
    )
    mwis_instances = tuple(default_mwis_instances())[
        : config["mwis_instance_count"]
    ]
    mwis = run_mwis_hybrid_benchmark(
        seeds=seeds,
        instances=mwis_instances,
        quantum_budget=config["mwis_budget"],
        bfdcqo_backend="pyqpanda",
        qaoa_implementation="upstream",
    )

    summaries = {
        "maxcut": _enrich_summary(
            maxcut, keys=("instance", "solver"), metric="p_opt"
        ),
        "max3sat": _enrich_summary(
            max3sat,
            keys=("instance", "solver"),
            metric="p_opt_logical",
        ),
        "family": _family_summary(family),
        "mwis": _enrich_summary(
            mwis, keys=("instance", "solver"), metric="p_opt"
        ),
    }
    payloads = {
        "maxcut": maxcut,
        "max3sat": max3sat,
        "family": family,
        "mwis": mwis,
    }
    for name, payload in payloads.items():
        _write_csv(output_dir / f"{name}_raw.csv", payload["rows"])
        _write_csv(output_dir / f"{name}_summary.csv", summaries[name])
        (output_dir / f"{name}_raw.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    metadata = _metadata(repository, config, quick)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"metadata": metadata, "payloads": payloads, "summaries": summaries}


def _range_text(values: Iterable[float], digits: int = 0) -> str:
    materialized = tuple(float(value) for value in values)
    formatter = f"{{:.{digits}f}}"
    low = formatter.format(min(materialized))
    high = formatter.format(max(materialized))
    return low if low == high else f"{low}–{high}"


def _write_figures(result: dict[str, object], docs_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = docs_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    maxcut = result["summaries"]["maxcut"]
    instances = sorted({row["instance"] for row in maxcut})
    solvers = ("BF-DCQO", "QAOA-p1-upstream")
    x = np.arange(len(instances))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for offset, solver in zip((-width / 2, width / 2), solvers):
        values = [
            next(
                float(row["mean_expected_approx_ratio"])
                for row in maxcut
                if row["instance"] == instance and row["solver"] == solver
            )
            for instance in instances
        ]
        ax.bar(x + offset, values, width, label=solver)
    ax.set_xticks(x, instances)
    ax.set_ylabel("Mean expected approximation ratio")
    ax.set_title("MaxCut on PyQPanda3 CPUQVM")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures / "maxcut_bf_vs_upstream_qaoa.png", dpi=180
    )
    plt.close(fig)

    mwis = result["summaries"]["mwis"]
    instances = sorted({row["instance"] for row in mwis})
    solvers = ("Hybrid-random", "Hybrid-confidence")
    x = np.arange(len(instances))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for offset, solver in zip((-width / 2, width / 2), solvers):
        values = [
            next(
                float(row["mean_weight_ratio"])
                for row in mwis
                if row["instance"] == instance and row["solver"] == solver
            )
            for instance in instances
        ]
        ax.bar(x + offset, values, width, label=solver)
    ax.set_xticks(x, instances)
    ax.set_ylabel("Mean best-weight ratio")
    ax.set_title("MWIS equal-cost hybrid ablation")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "mwis_hybrid_ablation.png", dpi=180)
    plt.close(fig)


def _write_report(result: dict[str, object], docs_dir: Path) -> None:
    metadata = result["metadata"]
    payloads = result["payloads"]
    maxcut_q = [
        row for row in payloads["maxcut"]["rows"]
        if row["solver"] == "QAOA-p1-upstream"
    ]
    max3sat_q = [
        row for row in payloads["max3sat"]["rows"]
        if row["solver"] == "QAOA-p1-upstream"
    ]
    mwis_q = [
        row for row in payloads["mwis"]["rows"]
        if row["solver"] == "QAOA-p1-upstream"
    ]
    max3_rows = payloads["max3sat"]["rows"]
    native = [row for row in max3_rows if row["formulation"] == "native-HUBO"]
    quadratized = [
        row for row in max3_rows
        if row["formulation"] == "Rosenberg-quadratized-QUBO"
    ]
    mwis_summary = result["summaries"]["mwis"]
    random_ratio = _mean(
        row["mean_weight_ratio"]
        for row in mwis_summary if row["solver"] == "Hybrid-random"
    )
    confidence_ratio = _mean(
        row["mean_weight_ratio"]
        for row in mwis_summary if row["solver"] == "Hybrid-confidence"
    )

    def budget_line(label: str, rows: list[dict[str, object]]) -> str:
        return (
            f"- {label}：实际 circuit evaluations "
            f"{_range_text(row['circuit_evaluations'] for row in rows)}，总 shots "
            f"{_range_text(row['total_shots'] for row in rows)}，目标预算 "
            f"{_range_text(row['target_shot_budget'] for row in rows)}，delta "
            f"{_range_text(row['budget_delta'] for row in rows)}。"
        )

    report = f"""# BF-DCQO CPUQVM 基准报告

本报告由 `scripts/run_bfdcqo_cpuqvm_benchmarks.py` 的 official 配置生成。软件版本为 pyqpanda-algorithm {metadata['package_version']}、PyQPanda3 {metadata['pyqpanda3_version']}、Python {metadata['python_version']}；记录的源码 commit 为 `{metadata['git_commit']}`，UTC 时间为 `{metadata['generated_at_utc']}`。

## 配置与证据边界

- `maxcut_summary.csv`：{metadata['instance_counts']['maxcut']} 个实例、{len(metadata['seeds'])} 个初值种子；BF-DCQO 使用 `{BF_BACKEND}`，QAOA 使用 `{QAOA_BACKEND}`。
- `max3sat_summary.csv`：{metadata['instance_counts']['max3sat']} 个受控小实例、{len(metadata['seeds'])} 个初值种子；BF-DCQO 使用 `{BF_BACKEND}`，QAOA 使用 `{QAOA_BACKEND}`。
- `family_summary.csv`：2 个受控问题、预算 {metadata['configuration']['family_budgets']}、{len(metadata['seeds'])} 个种子；全部使用 `{BF_BACKEND}`。
- `mwis_summary.csv`：{metadata['instance_counts']['mwis']} 个受控 8 顶点实例、{len(metadata['seeds'])} 个初值种子；BF/Hybrid 使用 `{BF_BACKEND}`，QAOA 使用 `{QAOA_BACKEND}`。
- `initial_parameter_seed` 只控制优化器初值；CPUQVM 的 shot sampling 未由该参数固定。因此结果是可复现配置下的真实采样运行，不应误写成逐 shot 确定性复现。

## 上游 QAOA 预算核算

上游 QAOA 的总 shots 不是按上限推定，而是以 `observed circuit_evaluations × shots` 记录；目标预算按 `(max_evals + 2) × shots` 定义。

{budget_line('MaxCut', maxcut_q)}
{budget_line('MAX-3-SAT', max3sat_q)}
{budget_line('MWIS', mwis_q)}

## 结果解读

MAX-3-SAT 的 native HUBO 使用 {_range_text(row['solver_n'] for row in native)} 个逻辑量子比特，而 Rosenberg 二次化基线使用 {_range_text(row['solver_n'] for row in quadratized)} 个量子比特（含 {_range_text(row['ancillas'] for row in quadratized)} 个辅助变量）。这个量子比特节省伴随更高权重 Pauli 旋转：native 路径的观测最大权重为 {_range_text(row['max_pauli_weight'] for row in native)}，二次化 QAOA 为 {_range_text(row['max_pauli_weight'] for row in quadratized)}，所以不能只凭 qubit 数判断总体资源更优。

MWIS 等成本消融中，随机分块 Hybrid 的跨实例平均 best-weight ratio 为 {random_ratio:.6f}，置信度引导 Hybrid 为 {confidence_ratio:.6f}。两者量子预算和局部枚举评估次数相同；图表只陈述这批受控实例的观测结果，不外推到其他图族。

所有实例都很小，允许精确求解做校验；种子数也仅为 {len(metadata['seeds'])}。这些数据不构成量子优势证据，也不支持 BF-DCQO 对 QAOA 或其他优化器普遍优越的结论。

![MaxCut BF-DCQO 与上游 QAOA](figures/maxcut_bf_vs_upstream_qaoa.png)

![MWIS Hybrid 等成本消融](figures/mwis_hybrid_ablation.png)

## 方法归属

- Basic BF-DCQO 的核心协议归于 A. Gomez Cadavid 等人的 *Bias-field digitized counterdiabatic quantum optimization*（Physical Review Research 7, L022010, 2025；arXiv:2405.13898）。
- native HUBO、CVaR BF-DCQO 与 MAX-3-SAT 构造归于 S. V. Romero 等人的 *Bias-Field Digitized Counterdiabatic Quantum Algorithm for Higher-Order Binary Optimization*（Communications Physics 8, 348, 2025；arXiv:2409.04477）。
- 经典 refinement 的先行概念参见 Hybrid Sequential Quantum Computing（HSQC）；这里的 MWIS 路径是受预算并记录评估次数的工程化消融，并非对 HSQC 的原创性主张。
"""
    (docs_dir / "BENCHMARK_REPORT.md").write_text(report, encoding="utf-8")


def write_curated_evidence(result: dict[str, object], docs_dir: Path) -> None:
    """Write only aggregated tables, two figures, and the Chinese report."""
    docs_dir = Path(docs_dir).resolve()
    tables = docs_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, rows in result["summaries"].items():
        _write_csv(tables / f"{name}_summary.csv", rows)
    _write_figures(result, docs_dir)
    _write_report(result, docs_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docs-output", type=Path)
    args = parser.parse_args()
    result = run_suite(args.output, quick=args.quick)
    if args.docs_output is not None:
        write_curated_evidence(result, args.docs_output)
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
