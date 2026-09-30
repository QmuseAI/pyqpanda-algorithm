# PyBF-DCQO Competition Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 BF-DCQO 数学实现的前提下，把交付包整合到最新 `upstream/develop`，完成真实 PyQPanda3 CPUQVM、上游 QAOA、全量测试、中文 README 与中文 PR 文案验证。

**Architecture:** 保留现有 `pyqpanda_alg/BFDCQO` 分层与 NumPy 独立参考后端，只在包入口、PyQPanda3 后端边界、benchmark 适配和文档层做上游集成。新增独立 `UpstreamQAOASolver` 调用仓库自带 `pyqpanda_alg.QAOA`，保留原 `QAOASolver` 作为明确标注的参考回归实现；官方 benchmark 通过参数选择 CPUQVM 与上游 QAOA，不改变已有数学回归默认值。

**Tech Stack:** Python 3.13、NumPy、SciPy、SymPy、PyQPanda3 0.3.5、pytest、Sphinx AutoAPI、Matplotlib、PowerShell、Git。

---

## 文件结构

### 新增或机械整合

- `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/`：交付包源码；复制时排除 `__pycache__` 和 `*.pyc`。
- `pyqpanda-algorithm/example/BFDCQO/`：四个可运行示例。
- `test/BFDCQO/Test_benchmark.py`
- `test/BFDCQO/Test_core.py`
- `test/BFDCQO/Test_family.py`
- `test/BFDCQO/Test_hubo.py`
- `test/BFDCQO/Test_max3sat.py`
- `test/BFDCQO/Test_mwis_hybrid.py`
- `test/BFDCQO/Test_solver.py`
- `test/BFDCQO/Test_public_api.py`：公开 API、docstring 与顶层导入检查。
- `test/BFDCQO/Test_cpuqvm_integration.py`：CPUQVM 位序、counts 与任意 Pauli 字符串验证。
- `test/BFDCQO/Test_upstream_qaoa.py`：Ising→SymPy 等价性与上游 QAOA 计量测试。
- `test/BFDCQO/Test_benchmark_driver.py`：正式 benchmark 驱动快速模式验证。
- `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/upstream_qaoa.py`：上游 QAOA 适配器。
- `scripts/verify_pyqpanda_cpuqvm.py`：无 fallback 的强制 CPUQVM 检查。
- `scripts/run_bfdcqo_cpuqvm_benchmarks.py`：正式/快速 CPUQVM benchmark 驱动。
- `docs/bfdcqo/BENCHMARK_REPORT.md`：中文 benchmark 摘要、限制和归属。
- `docs/bfdcqo/tables/*.csv`：精简 CPUQVM 表格。
- `docs/bfdcqo/figures/*.png`：少量正式图。

### 修改

- `pyqpanda-algorithm/pyqpanda_alg/__init__.py`：公开 `BFDCQO` 模块。
- `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/README.md`：改为中文为主。
- `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/backends/pyqpanda3_backend.py`：只允许 API/位序适配，不改数学。
- `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/__init__.py`：导出上游适配器。
- `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/runner.py`：MaxCut 支持 CPUQVM 与上游 QAOA。
- `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/max3sat.py`：MAX-3-SAT 支持 CPUQVM 与上游 QAOA。
- `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/family.py`：BF family 后端参数化。
- `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/mwis.py`：MWIS 后端与 QAOA 实现参数化。
- `test/pytest.ini`：把 `BFDCQO` 加入 `testpaths`。
- `README.md`：在中文优化算法索引加入 BF-DCQO。
- `Tutorials/source/index.rst`：把 AutoAPI 的 BFDCQO 页面加入导航。
- 外层交付包 `README.md`、`PR_DESCRIPTION.md`、`BENCHMARK_REPORT.md`、`SUBMISSION_VALIDATION_REPORT.md`：更新为本地实测后的中文版本；这些文件不进入上游 Git 分支。

### 数学冻结区

以下文件只能机械复制，不因集成需求改变公式或参考数值：

- `core/agp.py`
- `core/schedule.py`
- `core/feedback.py`
- `core/hamiltonian.py`
- `problems/ising.py`
- `problems/polynomial.py`
- `applications/max3sat.py`
- `applications/mwis.py`
- `solvers/bfdcqo.py`
- `local_search/confidence.py`

如果某一步需要修改这些文件中的数学表达、能量约定、反馈规则或优化逻辑，停止执行并向用户报告。

---

### Task 1: 干净整合现有 BF-DCQO 源码、示例与回归测试

**Files:**
- Create: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/**`
- Create: `pyqpanda-algorithm/example/BFDCQO/*.py`
- Create: `test/BFDCQO/Test_*.py`
- Modify: `test/pytest.ini`

- [ ] **Step 1: 证明上游基线尚无 BFDCQO 模块**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path "pyqpanda-algorithm").Path
python -c "import pyqpanda_alg.BFDCQO"
```

Expected: `ModuleNotFoundError: No module named 'pyqpanda_alg.BFDCQO'`。

- [ ] **Step 2: 从交付包机械复制源码和示例，排除缓存**

Run from the worktree root:

```powershell
$bundleRoot = (Resolve-Path "..\..\..").Path
$sourcePackage = Join-Path $bundleRoot "pyqpanda-algorithm\pyqpanda_alg\BFDCQO"
$targetPackage = Join-Path (Get-Location) "pyqpanda-algorithm\pyqpanda_alg\BFDCQO"
$sourceExamples = Join-Path $bundleRoot "pyqpanda-algorithm\example\BFDCQO"
$targetExamples = Join-Path (Get-Location) "pyqpanda-algorithm\example\BFDCQO"

Get-ChildItem -LiteralPath $sourcePackage -Recurse -File |
    Where-Object { $_.FullName -notmatch '__pycache__' -and $_.Extension -ne '.pyc' } |
    ForEach-Object {
        $relative = $_.FullName.Substring($sourcePackage.Length).TrimStart('\')
        $destination = Join-Path $targetPackage $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $destination
    }

Get-ChildItem -LiteralPath $sourceExamples -File -Filter '*.py' | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $targetExamples | Out-Null
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $targetExamples $_.Name)
}
```

Expected: `rg --files pyqpanda-algorithm/pyqpanda_alg/BFDCQO` 不包含 `__pycache__` 或 `.pyc`。

- [ ] **Step 3: 按仓库实际约定复制并重命名测试**

Run:

```powershell
$sourceTests = Join-Path $bundleRoot "pyqpanda-algorithm\test\BFDCQO"
$targetTests = Join-Path (Get-Location) "test\BFDCQO"
New-Item -ItemType Directory -Force -Path $targetTests | Out-Null
$mapping = @{
    'test_benchmark.py' = 'Test_benchmark.py'
    'test_core.py' = 'Test_core.py'
    'test_family.py' = 'Test_family.py'
    'test_hubo.py' = 'Test_hubo.py'
    'test_max3sat.py' = 'Test_max3sat.py'
    'test_mwis_hybrid.py' = 'Test_mwis_hybrid.py'
    'test_solver.py' = 'Test_solver.py'
}
foreach ($entry in $mapping.GetEnumerator()) {
    Copy-Item -LiteralPath (Join-Path $sourceTests $entry.Key) -Destination (Join-Path $targetTests $entry.Value)
}
```

- [ ] **Step 4: 将 BFDCQO 加入 pytest 收集路径**

Apply this exact change to `test/pytest.ini`:

```ini
testpaths =
    QAlgBase
    QAOA
    QRAM
    QPCA
    QSVM
    BFDCQO
```

- [ ] **Step 5: 运行复制后的数学与回归测试**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path "pyqpanda-algorithm").Path
python -m pytest test/BFDCQO -q -o "addopts="
```

Expected: `53 passed`，CPUQVM 测试因本机已安装 PyQPanda3 而不应跳过。

- [ ] **Step 6: 核对数学冻结区与交付包完全一致**

Run a SHA-256 comparison for every frozen file:

```powershell
$frozen = @(
  'core/agp.py','core/schedule.py','core/feedback.py','core/hamiltonian.py',
  'problems/ising.py','problems/polynomial.py','applications/max3sat.py',
  'applications/mwis.py','solvers/bfdcqo.py','local_search/confidence.py'
)
foreach ($file in $frozen) {
    $source = Join-Path $sourcePackage $file
    $target = Join-Path $targetPackage $file
    if ((Get-FileHash $source).Hash -ne (Get-FileHash $target).Hash) {
        throw "数学冻结文件不一致: $file"
    }
}
```

Expected: command exits 0 with no output。

- [ ] **Step 7: 提交机械整合**

```powershell
git add pyqpanda-algorithm/pyqpanda_alg/BFDCQO pyqpanda-algorithm/example/BFDCQO test/BFDCQO test/pytest.ini
git commit -m "feat: integrate BF-DCQO solver family"
```

---

### Task 2: 公开 API 与中文文档索引

**Files:**
- Create: `test/BFDCQO/Test_public_api.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/__init__.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/README.md`
- Modify: `README.md`
- Modify: `Tutorials/source/index.rst`

- [ ] **Step 1: 写公开 API 失败测试**

Create `test/BFDCQO/Test_public_api.py`:

```python
import importlib
import inspect
import subprocess
import sys

import pyqpanda_alg

BFDCQO = importlib.import_module("pyqpanda_alg.BFDCQO")


def test_bfdcqo_is_exposed_from_package_root():
    completed = subprocess.run(
        [sys.executable, "-c", "import pyqpanda_alg; assert hasattr(pyqpanda_alg, 'BFDCQO')"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_all_public_bfdcqo_symbols_have_docstrings():
    missing = []
    for name in BFDCQO.__all__:
        value = getattr(BFDCQO, name)
        if inspect.isclass(value) or inspect.isfunction(value):
            if not inspect.getdoc(value):
                missing.append(name)
    assert missing == []
```

- [ ] **Step 2: 运行测试并确认顶层导入失败**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path "pyqpanda-algorithm").Path
python -m pytest test/BFDCQO/Test_public_api.py -q -o "addopts="
```

Expected: collection fails because `pyqpanda_alg.__init__` has not exposed `BFDCQO`。

- [ ] **Step 3: 在包入口公开 BFDCQO**

Append to `pyqpanda-algorithm/pyqpanda_alg/__init__.py`:

```python
from . import BFDCQO
```

- [ ] **Step 4: 把模块 README 改为中文为主**

Keep the runnable unified-API and native-HUBO code examples unchanged, but use these exact Chinese headings and claims:

```markdown
# PyQPanda 的 PyBF-DCQO 算法族

`BFDCQO` 为 `pyqpanda-algorithm` 提供模块化的偏置场数字化反绝热量子优化算法族。

## 功能概览
## 统一 API
## 原生 HUBO
## 求解器变体
### 基础 BF-DCQO
### CVaR BF-DCQO
### 自适应采样 BF-DCQO
### Hybrid BF-DCQO
## 应用与 benchmark
## 正确性验证
## 适用范围与限制
## 参考文献与归属
```

The limitations section must state verbatim:

```markdown
当前实现定位为小规模算法验证、教学与 benchmark 参考实现。精确基态诊断和部分指标需要枚举完整计算基；在完成 count-only 与可选精确诊断路径之前，不应将其描述为可直接扩展到大规模量子硬件的求解器。
```

- [ ] **Step 5: 更新根 README 中文算法索引**

Insert after QAOA in `README.md`:

```markdown
- **BF-DCQO（偏置场数字化反绝热量子优化）**  
  面向 QUBO 与原生多项式 Ising/HUBO 问题，提供基础、CVaR、自适应采样与受预算约束的混合求解变体，适用于小规模组合优化验证与研究 benchmark。
```

- [ ] **Step 6: 更新 Sphinx AutoAPI 导航**

Add after the QAOA entry in `Tutorials/source/index.rst`:

```rst
    autoapi/pyqpanda_alg/BFDCQO/index
```

- [ ] **Step 7: 验证 API 和文档入口**

Run:

```powershell
python -m pytest test/BFDCQO/Test_public_api.py -q -o "addopts="
python -c "from pyqpanda_alg import BFDCQO; print(BFDCQO.BFDCQOSolver.__name__)"
```

Expected: tests pass and output is `BFDCQOSolver`。

- [ ] **Step 8: 提交 API 与中文索引**

```powershell
git add pyqpanda-algorithm/pyqpanda_alg/__init__.py pyqpanda-algorithm/pyqpanda_alg/BFDCQO/README.md README.md Tutorials/source/index.rst test/BFDCQO/Test_public_api.py
git commit -m "docs: add Chinese BF-DCQO API documentation"
```

---

### Task 3: 强化真实 PyQPanda3 CPUQVM 位序与旋转验证

**Files:**
- Create: `test/BFDCQO/Test_cpuqvm_integration.py`
- Create: `scripts/verify_pyqpanda_cpuqvm.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/backends/pyqpanda3_backend.py`（仅允许 CPUQVM API、门调用与位序边界修正；测试已通过时保持内容不变）

- [ ] **Step 1: 写 CPUQVM 集成测试**

Create `test/BFDCQO/Test_cpuqvm_integration.py` with these cases:

```python
import numpy as np

from pyqpanda_alg.BFDCQO.backends.numpy_statevector import (
    apply_pauli_rotation,
    product_state_from_ry,
)
from pyqpanda_alg.BFDCQO.backends.pyqpanda3_backend import (
    pauli_rotation_recipe,
    run_bfdcqo_probabilities,
)


def test_mixed_xyz_rotation_recipe_has_expected_parity_ladder():
    recipe = pauli_rotation_recipe(("X", "Y", "I", "Z"), 0.37)
    assert recipe == [
        ("H", 0), ("RX", 1, np.pi / 2),
        ("CNOT", 0, 3), ("CNOT", 1, 3), ("RZ", 3, 0.74),
        ("CNOT", 1, 3), ("CNOT", 0, 3),
        ("RX", 1, -np.pi / 2), ("H", 0),
    ]


def test_cpuqvm_mixed_xyz_rotation_matches_numpy_probabilities():
    angles = np.array([0.31, -0.42, 0.18, 0.77])
    pauli = ("X", "Y", "I", "Z")
    theta = 0.37
    state = product_state_from_ry(angles)
    expected = np.abs(apply_pauli_rotation(state, pauli, theta)) ** 2
    actual = run_bfdcqo_probabilities(4, angles, [[(pauli, theta)]])
    assert np.allclose(actual, expected, atol=1e-8)


def test_cpuqvm_probability_index_uses_q0_as_least_significant_bit():
    angles = np.array([np.pi, 0.0, np.pi])
    probability = run_bfdcqo_probabilities(3, angles, [])
    assert int(np.argmax(probability)) == 0b101
    assert np.isclose(probability[0b101], 1.0, atol=1e-8)
```

- [ ] **Step 2: 运行 CPUQVM 测试**

```powershell
$env:PYTHONPATH = (Resolve-Path "pyqpanda-algorithm").Path
python -m pytest test/BFDCQO/Test_cpuqvm_integration.py -q -o "addopts="
```

Expected: `3 passed`。If any failure implies a mathematical sign change rather than a backend gate-order/bit-order issue, stop and report instead of editing the frozen files。

- [ ] **Step 3: 复制并执行无 fallback 验证脚本**

Copy the existing outer `scripts/verify_pyqpanda_cpuqvm.py` into repository `scripts/verify_pyqpanda_cpuqvm.py`, then run:

```powershell
python scripts/verify_pyqpanda_cpuqvm.py
```

Expected:

```text
PASS: MWIS BF/QAOA PyQPanda3 CPUQVM probabilities agree with NumPy <= 1e-8
```

- [ ] **Step 4: 运行所有含 PyQPanda 的测试并检查无跳过**

```powershell
python -m pytest test/BFDCQO -q -rs -o "addopts=" | Tee-Object -FilePath cpuqvm-test-output.txt
if (Select-String -Path cpuqvm-test-output.txt -Pattern ' skipped') { throw 'CPUQVM tests were skipped' }
Remove-Item -LiteralPath cpuqvm-test-output.txt
```

Expected: all pass, no `skipped` summary。Delete the temporary text file after recording the result outside Git。

- [ ] **Step 5: 提交 CPUQVM 验证**

```powershell
git add test/BFDCQO/Test_cpuqvm_integration.py scripts/verify_pyqpanda_cpuqvm.py pyqpanda-algorithm/pyqpanda_alg/BFDCQO/backends/pyqpanda3_backend.py
git commit -m "test: validate BF-DCQO on PyQPanda3 CPUQVM"
```

---

### Task 4: 新增直接调用仓库上游 QAOA 的适配器

**Files:**
- Create: `test/BFDCQO/Test_upstream_qaoa.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/upstream_qaoa.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/__init__.py`

- [ ] **Step 1: 写 Ising→SymPy 与上游执行失败测试**

Create `test/BFDCQO/Test_upstream_qaoa.py`:

```python
import numpy as np

from pyqpanda_alg.BFDCQO.benchmark.upstream_qaoa import (
    UpstreamQAOASolver,
    ising_to_sympy,
)
from pyqpanda_alg.BFDCQO.problems import IsingProblem, MaxCutProblem


def test_ising_to_sympy_preserves_every_basis_energy():
    problem = IsingProblem(
        h=np.array([0.2, -0.4, 0.1]),
        J=np.array([[0.0, 0.7, 0.0], [0.7, 0.0, -0.3], [0.0, -0.3, 0.0]]),
        offset=0.6,
    )
    expression, symbols = ising_to_sympy(problem)
    for index in range(1 << problem.n):
        bits = [(index >> qubit) & 1 for qubit in range(problem.n)]
        actual = float(expression.subs(dict(zip(symbols, bits))))
        assert np.isclose(actual, problem.energy_bits(bits), atol=1e-12)


def test_upstream_qaoa_reports_observed_cpuqvm_budget():
    problem = MaxCutProblem.from_edges(3, [(0, 1), (1, 2), (0, 2)]).to_ising()
    result = UpstreamQAOASolver(
        problem,
        p=1,
        shots=32,
        max_evals=2,
        target_shot_budget=128,
        seed=7,
    ).solve()
    assert result.backend == "pyqpanda_alg.QAOA.CPUQVM"
    assert result.total_shots == result.circuit_evaluations * 32
    assert result.budget_delta == result.total_shots - 128
    assert np.isclose(sum(result.distribution.values()), 1.0, atol=1e-12)
    assert np.isclose(result.best_energy, problem.energy_bitstring(result.best_solution))
```

- [ ] **Step 2: 运行测试并确认模块缺失**

```powershell
python -m pytest test/BFDCQO/Test_upstream_qaoa.py -q -o "addopts="
```

Expected: collection fails because `benchmark.upstream_qaoa` does not exist。

- [ ] **Step 3: 实现能量等价转换**

In `benchmark/upstream_qaoa.py`, implement the exact expansion of
`z_i = 1 - 2 x_i`:

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import sympy as sp

from pyqpanda_alg.QAOA import qaoa as upstream_qaoa
from ..problems.ising import IsingProblem


def ising_to_sympy(problem: IsingProblem) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
    """Convert a quadratic Ising objective to an exactly equivalent binary polynomial."""
    symbols = tuple(sp.symbols(f"x0:{problem.n}"))
    expression = sp.Float(problem.offset)
    for index, field in enumerate(problem.h):
        expression += sp.Float(field) * (1 - 2 * symbols[index])
    for left in range(problem.n):
        for right in range(left + 1, problem.n):
            coupling = problem.J[left, right]
            if abs(coupling) > 1e-15:
                expression += sp.Float(coupling) * (1 - 2 * symbols[left]) * (1 - 2 * symbols[right])
    expression = sp.expand(expression)
    if expression.free_symbols != set(symbols):
        missing = sorted(str(symbol) for symbol in set(symbols) - expression.free_symbols)
        raise ValueError(f"upstream QAOA cannot represent unconstrained variables: {missing}")
    return expression, symbols
```

- [ ] **Step 4: 实现可审计的上游 QAOA 结果与求解器**

Add these fields and accounting rules to the same file:

```python
@dataclass(frozen=True)
class UpstreamQAOAResult:
    best_solution: str
    best_energy: float
    exact_ground_energy: float
    exact_ground_states: list[str]
    expected_energy: float
    ground_state_probability: float
    parameters: list[float]
    objective_evaluations: int
    circuit_evaluations: int
    shots_per_circuit: int
    total_shots: int
    target_shot_budget: int | None
    budget_delta: int | None
    distribution: dict[str, float]
    backend: str = "pyqpanda_alg.QAOA.CPUQVM"

    @property
    def success(self) -> bool:
        return bool(np.isclose(self.best_energy, self.exact_ground_energy, atol=1e-9))


class UpstreamQAOASolver:
    """Auditable benchmark adapter around the repository's public QAOA API."""

    def __init__(self, problem: IsingProblem, *, p: int = 1, shots: int = 256,
                 max_evals: int = 22, target_shot_budget: int | None = None,
                 seed: int = 7) -> None:
        if p < 1 or shots < 1 or max_evals < 1:
            raise ValueError("p, shots, and max_evals must be positive")
        self.problem = problem
        self.p = int(p)
        self.shots = int(shots)
        self.max_evals = int(max_evals)
        self.target_shot_budget = target_shot_budget
        self.seed = int(seed)

    def solve(self) -> UpstreamQAOAResult:
        expression, _ = ising_to_sympy(self.problem)
        solver = upstream_qaoa.QAOA(expression)
        random_state = np.random.get_state()
        try:
            np.random.seed(self.seed)
            distribution, parameters, _ = solver.run(
                layer=self.p,
                shots=self.shots,
                optimizer="COBYLA",
                optimizer_option={"options": {"maxiter": self.max_evals}},
            )
        finally:
            np.random.set_state(random_state)

        cleaned = {str(bits).replace(" ", ""): float(prob) for bits, prob in distribution.items()}
        total_probability = sum(cleaned.values())
        if total_probability <= 0:
            raise RuntimeError("upstream QAOA returned an empty distribution")
        cleaned = {bits: prob / total_probability for bits, prob in cleaned.items()}
        supported = [bits for bits, prob in cleaned.items() if prob > 0]
        best = min(supported, key=self.problem.energy_bitstring)
        exact_energy, exact_states = self.problem.exact_ground_states()
        expected = sum(prob * self.problem.energy_bitstring(bits) for bits, prob in cleaned.items())
        p_opt = sum(cleaned.get(bits, 0.0) for bits in exact_states)
        objective_evaluations = max(int(solver.iter) - 1, 0)
        circuit_evaluations = int(solver.circuit_iter)
        total_shots = circuit_evaluations * self.shots
        budget_delta = None if self.target_shot_budget is None else total_shots - int(self.target_shot_budget)
        return UpstreamQAOAResult(
            best_solution=best,
            best_energy=self.problem.energy_bitstring(best),
            exact_ground_energy=exact_energy,
            exact_ground_states=exact_states,
            expected_energy=float(expected),
            ground_state_probability=float(p_opt),
            parameters=[float(value) for value in parameters],
            objective_evaluations=objective_evaluations,
            circuit_evaluations=circuit_evaluations,
            shots_per_circuit=self.shots,
            total_shots=total_shots,
            target_shot_budget=self.target_shot_budget,
            budget_delta=budget_delta,
            distribution=cleaned,
        )
```

- [ ] **Step 5: 导出并验证上游适配器**

Export `UpstreamQAOASolver`, `UpstreamQAOAResult`, and `ising_to_sympy` from `benchmark/__init__.py`, then run:

```powershell
python -m pytest test/BFDCQO/Test_upstream_qaoa.py -q -o "addopts="
```

Expected: `2 passed`。

- [ ] **Step 6: 提交上游 QAOA 适配器**

```powershell
git add pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/upstream_qaoa.py pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/__init__.py test/BFDCQO/Test_upstream_qaoa.py
git commit -m "feat: benchmark with upstream PyQPanda QAOA"
```

---

### Task 5: 参数化 benchmark 后端并保留参考 QAOA 回归路径

**Files:**
- Modify: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/runner.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/max3sat.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/family.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark/mwis.py`
- Modify: `test/BFDCQO/Test_benchmark.py`
- Modify: `test/BFDCQO/Test_family.py`
- Modify: `test/BFDCQO/Test_max3sat.py`
- Modify: `test/BFDCQO/Test_mwis_hybrid.py`

- [ ] **Step 1: 写 benchmark 选择上游 QAOA 的失败测试**

Add a one-seed quick test to `Test_benchmark.py`:

```python
def test_maxcut_benchmark_can_use_cpuqvm_and_upstream_qaoa():
    payload = run_maxcut_benchmark(
        seeds=(3,),
        bfdcqo_backend="pyqpanda",
        qaoa_implementation="upstream",
        qaoa_shots=32,
        qaoa_max_evals=2,
    )
    rows = payload["rows"]
    assert {row["backend"] for row in rows if row["solver"] == "BF-DCQO"} == {"pyqpanda"}
    qaoa_rows = [row for row in rows if row["solver"] == "QAOA-p1-upstream"]
    assert qaoa_rows
    assert all(row["backend"] == "pyqpanda_alg.QAOA.CPUQVM" for row in qaoa_rows)
    assert all("budget_delta" in row for row in qaoa_rows)
```

- [ ] **Step 2: 运行测试并确认新参数尚未支持**

```powershell
python -m pytest test/BFDCQO/Test_benchmark.py::test_maxcut_benchmark_can_use_cpuqvm_and_upstream_qaoa -q -o "addopts="
```

Expected: fails with unexpected keyword arguments。

- [ ] **Step 3: 为 BF benchmark 增加后端参数，默认保持回归值不变**

Use these exact defaults to protect existing tests:

```python
def run_maxcut_benchmark(
    *, seeds=(3, 7, 11), bfdcqo_backend="numpy",
    qaoa_implementation="reference", qaoa_backend="auto",
    qaoa_shots=256, qaoa_max_evals=20, output_dir=None,
):
    ...

def run_family_benchmark(
    *, budgets=(1536, 3072, 6144), seeds=tuple(range(20)),
    cvar_alpha=0.10, backend="numpy", output_dir=None,
):
    ...

def run_adaptive_efficiency_benchmark(
    *, budget_caps=(1536, 3072, 6144), seeds=tuple(range(20)),
    cvar_alpha=0.10, backend="numpy", output_dir=None,
):
    ...
```

Every BF solver construction must use the passed `backend`. Payload configuration must report the actual backend rather than a hard-coded `numpy_reference` label。

- [ ] **Step 4: 在三个应用 benchmark 中选择 QAOA 实现**

For `qaoa_implementation == "upstream"`, construct:

```python
qaoa = UpstreamQAOASolver(
    problem,
    p=1,
    shots=qaoa_shots,
    max_evals=qaoa_max_evals,
    target_shot_budget=target_budget,
    seed=seed,
).solve()
```

Label it `QAOA-p1-upstream`, store `budget_delta`, and compute metrics from its returned `distribution`. For `qaoa_implementation == "reference"`, retain the current `QAOASolver` path and label it `QAOA-p1-reference`. Reject any other value with:

```python
raise ValueError("qaoa_implementation must be 'upstream' or 'reference'")
```

- [ ] **Step 5: 为 MAX-3-SAT 和 MWIS 增加快速上游 smoke tests**

Use one instance/one seed through optional `instances` or slice parameters so tests do not run the full suite. Assert:

```python
assert row["solver"] == "QAOA-p1-upstream"
assert row["backend"] == "pyqpanda_alg.QAOA.CPUQVM"
assert row["total_shots"] > 0
assert abs(row["budget_delta"]) <= row["total_shots"]
```

- [ ] **Step 6: 运行所有 benchmark 回归**

```powershell
python -m pytest test/BFDCQO/Test_benchmark.py test/BFDCQO/Test_family.py test/BFDCQO/Test_max3sat.py test/BFDCQO/Test_mwis_hybrid.py -q -o "addopts="
```

Expected: all existing reference regressions and new upstream smoke tests pass。

- [ ] **Step 7: 提交 benchmark 集成**

```powershell
git add pyqpanda-algorithm/pyqpanda_alg/BFDCQO/benchmark test/BFDCQO
git commit -m "feat: run BF-DCQO benchmarks on CPUQVM"
```

---

### Task 6: 增加可复现 CPUQVM benchmark 驱动并生成精简证据

**Files:**
- Create: `scripts/run_bfdcqo_cpuqvm_benchmarks.py`
- Create: `test/BFDCQO/Test_benchmark_driver.py`
- Create: `docs/bfdcqo/tables/*.csv`
- Create: `docs/bfdcqo/figures/*.png`
- Create: `docs/bfdcqo/BENCHMARK_REPORT.md`

- [ ] **Step 1: 写 benchmark 驱动的快速模式测试**

The script must expose:

```python
def run_suite(output_dir: Path, *, quick: bool) -> dict[str, object]:
    ...
```

Add a subprocess smoke test that runs `--quick` into pytest `tmp_path` and asserts these files exist:

```text
maxcut_summary.csv
max3sat_summary.csv
family_summary.csv
mwis_summary.csv
run_metadata.json
```

Use this test body in `test/BFDCQO/Test_benchmark_driver.py`:

```python
import os
from pathlib import Path
import subprocess
import sys


def test_cpuqvm_benchmark_driver_quick(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    script = repository / "scripts" / "run_bfdcqo_cpuqvm_benchmarks.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "pyqpanda-algorithm")
    completed = subprocess.run(
        [sys.executable, str(script), "--quick", "--output", str(tmp_path)],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = {
        "maxcut_summary.csv",
        "max3sat_summary.csv",
        "family_summary.csv",
        "mwis_summary.csv",
        "run_metadata.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
```

- [ ] **Step 2: 运行测试并确认驱动尚不存在**

```powershell
python -m pytest test/BFDCQO -k benchmark_driver -q -o "addopts="
```

Expected: failure because the script is absent。

- [ ] **Step 3: 实现 quick 与 official 两组固定配置**

Use these configurations:

```python
QUICK = {
    "seeds": (0,),
    "family_budgets": (1536,),
    "maxcut_qaoa_shots": 256,
    "maxcut_qaoa_max_evals": 22,
    "mwis_budget": 72,
}

OFFICIAL = {
    "seeds": (0, 1, 2),
    "family_budgets": (1536, 3072, 6144),
    "maxcut_qaoa_shots": 256,
    "maxcut_qaoa_max_evals": 22,
    "mwis_budget": 72,
}
```

All BF calls must use `backend="pyqpanda"`; MaxCut, MAX-3-SAT and MWIS QAOA calls must use `qaoa_implementation="upstream"`. `run_metadata.json` must include Python version, PyQPanda3 version, git commit, UTC timestamp, seeds, configuration and the statement that upstream QAOA shot totals are observed rather than assumed。

- [ ] **Step 4: 运行 quick benchmark**

```powershell
$bundleRoot = (Resolve-Path "..\..\..").Path
$output = Join-Path $bundleRoot "validation\bfdcqo_cpuqvm_quick"
python scripts/run_bfdcqo_cpuqvm_benchmarks.py --quick --output $output
```

Expected: all five files are written and each CSV has at least one data row。

- [ ] **Step 5: 运行缩减种子正式 benchmark**

```powershell
$bundleRoot = (Resolve-Path "..\..\..").Path
$output = Join-Path $bundleRoot "validation\bfdcqo_cpuqvm_official"
python scripts/run_bfdcqo_cpuqvm_benchmarks.py --output $output
```

Expected: MaxCut、MAX-3-SAT、BF family 和 72-shot MWIS results complete without NumPy backend labels in official BF rows。

- [ ] **Step 6: 生成精简表格与图**

Copy only aggregated CSVs into `docs/bfdcqo/tables/`. Generate these two figures at 180 DPI:

```text
maxcut_bf_vs_upstream_qaoa.png
mwis_hybrid_ablation.png
```

Do not commit raw per-circuit logs, JSON dumps, caches, or figures beyond the two listed above。

- [ ] **Step 7: 编写中文 benchmark 报告**

`docs/bfdcqo/BENCHMARK_REPORT.md` must contain:

1. PyQPanda3 CPUQVM version and git commit。
2. Each table's exact backend and seed count。
3. Upstream QAOA actual circuit evaluations, total shots, target budget and delta。
4. Native HUBO qubit saving and higher-weight rotation tradeoff。
5. MWIS confidence-guided versus equal-cost random baseline。
6. Small-instance limitation and no quantum-advantage/general-superiority claim。
7. Cadavid、Romero 和 HSQC attribution。

- [ ] **Step 8: 提交 benchmark 驱动与精简证据**

```powershell
git add scripts/run_bfdcqo_cpuqvm_benchmarks.py test/BFDCQO/Test_benchmark_driver.py docs/bfdcqo
git commit -m "docs: add CPUQVM benchmark evidence"
```

---

### Task 7: 验证示例、文档构建与公共 API

**Files:**
- Modify only for integration defects: `pyqpanda-algorithm/example/BFDCQO/*.py`
- Modify only for doc defects: `pyqpanda-algorithm/pyqpanda_alg/BFDCQO/README.md`

- [ ] **Step 1: 运行四个示例的缩减 smoke 配置**

Use their callable functions or CLI parameters to reduce seeds while preserving the real CPUQVM path:

```powershell
python pyqpanda-algorithm/example/BFDCQO/basic_bfdcqo.py
python pyqpanda-algorithm/example/BFDCQO/family_benchmark.py --quick
python pyqpanda-algorithm/example/BFDCQO/max3sat_native_hubo.py --quick
python pyqpanda-algorithm/example/BFDCQO/mwis_hybrid.py --quick
```

Expected: each exits 0 and prints its backend. If the copied examples lack `--quick`, add `argparse` flags that only change run size/backend selection, not solver mathematics。

- [ ] **Step 2: 审计所有公开 API docstring**

```powershell
python -c "import inspect; from pyqpanda_alg import BFDCQO; missing=[n for n in BFDCQO.__all__ if (inspect.isclass(getattr(BFDCQO,n)) or inspect.isfunction(getattr(BFDCQO,n))) and not inspect.getdoc(getattr(BFDCQO,n))]; print(missing); raise SystemExit(bool(missing))"
```

Expected: `[]`。

- [ ] **Step 3: 构建 Sphinx AutoAPI 文档**

```powershell
python -m sphinx -b html Tutorials/source validation/sphinx-html
```

Expected: exit 0 and `validation/sphinx-html/autoapi/pyqpanda_alg/BFDCQO/index.html` exists。

- [ ] **Step 4: 提交示例/文档修复（仅在存在集成缺陷时）**

```powershell
git add pyqpanda-algorithm/example/BFDCQO pyqpanda-algorithm/pyqpanda_alg/BFDCQO/README.md Tutorials/source/index.rst
git diff --cached --quiet || git commit -m "docs: verify BF-DCQO examples and API navigation"
```

---

### Task 8: 全量回归、静态检查与仓库卫生

**Files:**
- Modify only when fixing integration defects outside the mathematical frozen area.

- [ ] **Step 1: 运行 BF-DCQO 全量测试**

```powershell
$env:PYTHONPATH = (Resolve-Path "pyqpanda-algorithm").Path
python -m pytest test/BFDCQO -q -o "addopts="
```

Expected: all pass, zero skipped。

- [ ] **Step 2: 运行上游完整测试**

```powershell
python -m pytest test -q -o "addopts="
```

Expected: all pass. Record the existing optional `gmpy2` Windows `0xc0000139` stderr warning separately if it reappears; the command must still return exit 0。

- [ ] **Step 3: 运行语法与静态检查**

```powershell
python -m compileall -q pyqpanda-algorithm/pyqpanda_alg/BFDCQO pyqpanda-algorithm/example/BFDCQO scripts
python -m mypy pyqpanda-algorithm/pyqpanda_alg/BFDCQO --ignore-missing-imports
```

Expected: compileall exit 0. For mypy, fix errors introduced by this contribution; record unrelated existing configuration limitations without changing unrelated modules。

- [ ] **Step 4: 审计缓存、密钥、本地路径和大文件**

```powershell
if (git ls-files | Select-String -Pattern '__pycache__|\.pyc$|allure-results|validation/') { throw 'generated files are tracked' }
if (git grep -n -I -E 'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|api[_-]?key\s*=|C:\\Users\\Lenovo') { throw 'sensitive or local data found' }
git ls-files | ForEach-Object {
    $item = Get-Item -LiteralPath $_ -ErrorAction SilentlyContinue
    if ($item -and $item.Length -gt 5MB) { throw "large tracked file: $_" }
}
git diff --check upstream/develop...HEAD
```

Expected: no generated cache, secret, local absolute path, file larger than 5 MiB, or whitespace error。

- [ ] **Step 5: 再次核对数学冻结区无修改**

Compare each frozen file against the original delivery copy exactly as in Task 1. Any difference requires stopping for user review。

- [ ] **Step 6: 提交纯集成修复（仅在存在时）**

```powershell
git add -u
git diff --cached --quiet || git commit -m "fix: resolve BF-DCQO integration issues"
```

---

### Task 9: 更新中文交付材料和 PR 文案（不推送）

**Files outside the Git repository:**
- Modify: `../../../README.md`
- Modify: `../../../PR_DESCRIPTION.md`
- Modify: `../../../BENCHMARK_REPORT.md`
- Modify: `../../../SUBMISSION_VALIDATION_REPORT.md`

- [ ] **Step 1: 把外层 README 改为中文主文档**

Use these sections:

```markdown
# PyBF-DCQO 竞赛 PR 准备包
## 已完成内容
## 本地验证结果
## 目录说明
## 使用与复现
## 适用范围与限制
## 提交前步骤
```

Include actual test counts, PyQPanda3 version, branch name and exact benchmark backend from Task 6。

- [ ] **Step 2: 生成中文 PR 标题与正文**

Use title:

```text
〖创新应用〗新增 PyBF-DCQO 算法族：原生 HUBO、自适应采样与组合优化应用
```

The body must include: change summary, user API example, CPUQVM verification, upstream QAOA budget table, benchmark caveats, attribution, test commands/results, scalability limitation, and `Relates to #13`。Do not use “首次”“量子优势” or an unqualified “优于 QAOA” claim。

- [ ] **Step 3: 更新 benchmark 与验证报告**

Copy only verified numbers from `docs/bfdcqo/tables/` and test outputs. Every table must state `PyQPanda3 CPUQVM`, seed count and whether QAOA shots are observed or target values。

- [ ] **Step 4: 中文材料自检**

```powershell
rg -n "first|novel|quantum advantage|beats QAOA|首次|量子优势|普遍优于" ..\..\..\README.md ..\..\..\PR_DESCRIPTION.md ..\..\..\BENCHMARK_REPORT.md
```

Expected: no overclaim wording except explicit negations/cautions。

---

### Task 10: 同步最新 develop 并准备用户审阅包

**Files:**
- Remove from final PR diff: `docs/superpowers/specs/2026-09-30-pybf-dcqo-competition-integration-design.md`
- Remove from final PR diff: `docs/superpowers/plans/2026-09-30-pybf-dcqo-competition-integration.md`

- [ ] **Step 1: 获取最新上游状态**

```powershell
git fetch upstream develop
git log --oneline HEAD..upstream/develop
```

Expected: no output if upstream has not moved. If it has moved, rebase locally:

```powershell
git rebase upstream/develop
```

If a conflict touches the mathematical frozen area, abort the rebase and report to the user rather than resolving it silently。

- [ ] **Step 2: 从最终 PR 文件差异中移除内部计划文档**

```powershell
git rm docs/superpowers/specs/2026-09-30-pybf-dcqo-competition-integration-design.md
git rm docs/superpowers/plans/2026-09-30-pybf-dcqo-competition-integration.md
git commit -m "chore: remove internal integration notes"
```

Expected: `git diff --name-only upstream/develop...HEAD` does not list `docs/superpowers/`。

- [ ] **Step 3: 重跑最终验证**

```powershell
$env:PYTHONPATH = (Resolve-Path "pyqpanda-algorithm").Path
python scripts/verify_pyqpanda_cpuqvm.py
python -m pytest test/BFDCQO -q -o "addopts="
python -m pytest test -q -o "addopts="
python -m compileall -q pyqpanda-algorithm/pyqpanda_alg/BFDCQO pyqpanda-algorithm/example/BFDCQO scripts
git diff --check upstream/develop...HEAD
git status --short --branch
```

Expected: CPUQVM PASS; all tests pass with zero BF-DCQO skips; compileall and diff-check exit 0; working tree clean。

- [ ] **Step 4: 核对远端与分支，不执行写操作**

```powershell
git remote -v
git branch --show-current
```

Expected:

```text
origin   https://github.com/QmuseAI/pyqpanda-algorithm.git
upstream https://github.com/OriginQ/pyqpanda-algorithm.git
feature/pybf-dcqo-competition
```

Do not run `git push`, `gh pr create`, browser PR submission, or any other remote write。

- [ ] **Step 5: 交付用户审阅信息**

Provide:

1. local worktree absolute path and branch name;
2. commits since `upstream/develop`;
3. changed-file summary and mathematical frozen-file hash result;
4. CPUQVM, BF-DCQO, upstream full-suite, compileall, mypy and docs results;
5. benchmark tables/figures and exact caveats;
6. clickable Chinese `README.md` and `PR_DESCRIPTION.md` paths;
7. confirmation that no remote push or PR creation occurred;
8. any remaining limitation or environment warning。
