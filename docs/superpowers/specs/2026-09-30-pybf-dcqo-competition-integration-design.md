# PyBF-DCQO 竞赛提交整合设计

## 1. 背景与目标

本设计用于把现有 PyBF-DCQO 交付包安全整合到 `OriginQ/pyqpanda-algorithm` 最新 `develop` 分支，并在本地完成 PyQPanda3 CPUQVM、上游 QAOA 对照、仓库测试和中文文档验证。

本轮只准备本地功能分支、提交记录和中文 PR 文案。未经用户再次确认，不向任何远端推送，也不创建 Pull Request。后续如需推送，只允许使用 `QmuseAI/pyqpanda-algorithm` fork，并以 `OriginQ/pyqpanda-algorithm:develop` 为 PR 目标。

## 2. 基线与分支策略

- 功能分支名：`feature/pybf-dcqo-competition`。
- 基线：`upstream/develop`，设计时对应提交 `5f973ef`。
- `origin` 固定指向 `https://github.com/QmuseAI/pyqpanda-algorithm.git`。
- `upstream` 固定指向 `https://github.com/OriginQ/pyqpanda-algorithm.git`。
- 使用仓库内 `.worktrees/pybf-dcqo-competition` 作为隔离工作区。
- 合入前后分别记录全量测试结果，避免把上游既有问题误判为本次回归。

## 3. 范围

### 3.1 本轮包含

1. 将交付包的 `pyqpanda_alg/BFDCQO` 模块整合到上游实际包结构。
2. 整合 BF-DCQO 示例与测试，并按仓库可实际收集的 `Test_*.py` 方式组织测试。
3. 保持 `BFDCQO` 大写缩写命名，与现有 `QAOA`、`QUBO` 等模块风格一致。
4. 增加面向上游 `pyqpanda_alg.QAOA` 的 benchmark 适配层。
5. 在真实 PyQPanda3 CPUQVM 上验证位序、Pauli 旋转分解和 BF/QAOA 概率结果。
6. 在 CPUQVM 上重新生成精简 benchmark 证据，并准确标注后端、种子数和预算口径。
7. 增加以中文为主的模块 README、仓库索引说明、验证报告和 PR 文案。
8. 运行上游测试、BF-DCQO 测试、语法编译、可用静态检查和 `git diff --check`。

### 3.2 本轮不包含

1. 不修改 BF-DCQO 的 AGP、调度、反馈、CVaR、HUBO/QUBO 编码等数学公式。
2. 不把当前实现宣传为可扩展到大规模硬件的问题求解器。
3. 不在本轮重构完整的 count-only、exact-optional 执行架构；这会显著改变 solver 控制流，应作为单独工作评审。
4. 不删除或覆盖与本贡献无关的上游代码。
5. 不执行远端推送、PR 创建或其他 GitHub 写操作。

如果 CPUQVM 或上游 QAOA 适配暴露的问题只能通过修改数学定义或数值算法解决，立即停止实现并向用户说明原因、影响和候选方案。

## 4. 代码结构与职责

### 4.1 BF-DCQO 模块

`pyqpanda-algorithm/pyqpanda_alg/BFDCQO/` 保持独立模块，并沿用交付包的清晰分层：

- `api.py`：统一的 `BFDCQOConfig` 与 `BFDCQOSolver` 用户入口。
- `core/`：Pauli 代数、AGP、调度、测量和反馈等数学核心。
- `problems/`：二次 Ising/QUBO 与稀疏多项式 Ising/HUBO 表示。
- `backends/`：NumPy 参考后端和 PyQPanda3 CPUQVM 适配。
- `solvers/`：BF-DCQO 家族、Exact 基线、参考 QAOA 和 hybrid 求解器。
- `applications/`：MaxCut、MWIS 和 weighted MAX-3-SAT 建模。
- `benchmark/`：可复现 benchmark 驱动、指标和实例。
- `local_search/`：受预算约束的 confidence-guided/random 局部搜索。

数学核心不因仓库适配而改变。PyQPanda3 API 差异仅在 `backends/pyqpanda3_backend.py` 内消化。

### 4.2 上游 QAOA 适配

新增单独的上游 QAOA benchmark 适配器，直接调用：

```python
from pyqpanda_alg.QAOA import qaoa

solver = qaoa.QAOA(sympy_binary_objective)
result = solver.run(...)
```

适配器负责：

1. 将同一 QUBO/Ising benchmark 问题转换为上游 QAOA 接受的 SymPy 二进制目标函数。
2. 固定随机种子、层数、shots 和优化器选项。
3. 记录上游 API 可观测的目标函数调用次数与实际采样配置。
4. 将上游返回的 bitstring 概率字典转换为统一 benchmark 指标。
5. 明确区分“上游 QAOA”与交付包中的“参考 QAOA 回归后端”。

上游 QAOA 若无法暴露精确的总 shots 消耗，不通过修改其算法伪造严格匹配；改为报告可审计的配置、目标函数调用次数及预算估计，并在 PR 中解释限制。

## 5. 数据流

### 5.1 BF-DCQO

问题实例先转换为统一 Ising/HUBO 表示，数学核心据此生成插值 Hamiltonian 和一阶 gauge potential；后端只执行对应线路。NumPy 后端输出参考概率，PyQPanda3 后端输出 CPUQVM 概率或 counts。反馈、解码和 benchmark 指标在公共层完成。

### 5.2 上游 QAOA

同一应用实例先生成二进制目标函数，再交给上游 `QAOA` 优化和采样。适配器只做输入、结果与计量口径转换，不替换或改写上游 QAOA 的优化逻辑。

### 5.3 Benchmark 产物

正式产物只保留维护者容易审核的复现脚本、精简 CSV 表格、少量 PNG 图和中文摘要。原始重复产物、大体积缓存和本地路径不进入提交。

## 6. 错误处理与风险控制

- PyQPanda3 不可导入或 API 不兼容：测试明确失败，不回退到 NumPy 冒充 CPUQVM 结果。
- 位序不一致：使用小规模已知态、概率字典与 counts 三方交叉验证，统一在后端边界转换。
- 任意 Pauli 字符串旋转不一致：分别覆盖 X、Y、Z 和混合高权重字符串，比较 CPUQVM 与独立 NumPy 状态向量。
- 上游 QAOA 预算不可精确统计：报告限制，不宣称严格 matched-shot。
- benchmark 耗时过长：使用 checklist 允许的缩减种子集，并在表格与报告中显式标注。
- 上游既有测试出现环境噪声：同时记录退出码、通过数量和日志，不将其隐藏或误归因于本贡献。
- 发现需要修改数学实现：停止并请求用户决策。

## 7. 测试设计

### 7.1 原始数学与回归测试

保留交付包中已经通过的 AGP、Pauli、QUBO/HUBO、CVaR、adaptive-shot、MAX-3-SAT、MWIS 和 hybrid 测试，避免仓库适配改变参考数值。

### 7.2 CPUQVM 测试

- 强制运行 `scripts/verify_pyqpanda_cpuqvm.py`，无 NumPy fallback。
- CPUQVM 与 NumPy 概率对照误差阈值保持 `1e-8`。
- CPUQVM 相关测试不得因本机已安装 PyQPanda3 而跳过。
- 覆盖 `QProg`、测量字符串、`get_counts()`、`get_prob_dict()` 的位序。

### 7.3 上游集成测试

- 验证 BF-DCQO 顶层导入、公开 API 和 docstring。
- 验证上游 QAOA 适配器使用的确为 `pyqpanda_alg.QAOA`。
- 验证同一二进制目标函数在 Exact、BF-DCQO 与上游 QAOA 指标层采用一致能量约定。
- 运行 `test/` 下的全部上游测试，并清楚记录任何既有环境告警。

### 7.4 最终检查

- `python -m pytest`：BF-DCQO 与上游完整测试。
- `python -m compileall`：新增 Python 源码和示例。
- 项目现有 formatter、linter 或静态检查（如存在）。
- 公共 API docstring 审计。
- 缓存、密钥、本地绝对路径和大文件审计。
- `git diff --check`。

## 8. 文档与表述

- README、模块说明、验证报告和 PR 正文以中文为主。
- API 名称、论文题名和必要技术术语保留英文。
- Basic BF-DCQO 归属 Cadavid 等人的工作。
- Native HUBO、CVaR 和 MAX-3-SAT 归属 Romero 等人的工作。
- MWIS 明确为原始 BF-DCQO 论文已有应用。
- 描述经典 refinement 时提及 HSQC 的先行概念。
- 不使用“首次”“量子优势”或无条件“BF-DCQO 优于 QAOA”等表述。

## 9. 验收标准

本轮完成需同时满足：

1. 本地分支基于最新 `upstream/develop`，且不包含无关上游覆盖。
2. BF-DCQO 模块、示例、测试和中文文档正确进入上游结构。
3. 强制 CPUQVM 验证通过，CPUQVM 测试无跳过。
4. 上游 QAOA 适配和至少缩减种子集的 CPUQVM benchmark 可复现。
5. 上游原有测试和 BF-DCQO 测试通过，或对上游既有环境问题提供可复核证据。
6. README 与 PR 文案准确说明小规模模拟器限制和 benchmark 口径。
7. `git diff --check`、缓存/密钥/大文件审计通过。
8. 未向远端推送，未创建 PR；用户确认后才进入远端提交阶段。
