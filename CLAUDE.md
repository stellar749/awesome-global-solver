# Project: global_optim — C++ Global Optimization Solver Library

## User
- AD/robotics C++ engineer, working on mipilot project (Bazel-based autonomous driving)
- Interested in stochastic global optimization algorithms for trajectory planning
- Communicates in Chinese

## Project Status
- **Phase:** Phase 2 完成（CMA-ES / xNES / MPPI / SVGD + L1 验证），准备 Phase 3
- **Spec Doc:** `docs/spec.md`（完整的项目规格文档）
- **Memory:** `docs/memory.md`（项目记忆副本）
- **Implementation Notes:** `docs/implementation_notes.md`（每 Phase 的 bug / 设计决策记录）

## Tech Stack
- C++17, Eigen (线性代数), CMake (构建), Google Test (测试)

## Algorithms (V1)
1. **CMA-ES** — 协方差矩阵自适应进化策略 (IGO framework)
2. **xNES** — 指数自然进化策略
3. **SVGD** — Stein 变分梯度下降（需要梯度，天然多解发现）
4. **MPPI** — 模型预测路径积分控制（轨迹优化专用接口）

## Algorithms (V2 扩展)
5. **REINFORCE** — 策略梯度基线（线性高斯策略，C++）
6. **Flow Matching** — 条件流匹配（PyTorch，amortized trajectory optimization）

## Architecture
- `Problem` 统一接口 — 所有优化问题的统一抽象（可选梯度，for SVGD）
- `ControlSequenceProblem` — 轨迹问题的控制序列参数化（dynamics rollout）
- `ViaPointProblem` — 轨迹问题的 via-point 降维参数化（可选）
- `Solver` 统一基类 — 所有算法（含 MPPI）共享同一接口
- 所有算法统一最小化

## Validation (三层)
- **L1 正确性:** Sphere, Ellipsoid, Rosenbrock, Cigar — 确认实现正确
- **L2 全局优化:** Rastrigin, Double-Rosenbrock(双漏斗), Gaussian Mixture(多解发现), Random-Basin — 非凸多模态
- **L3 应用场景:** 2D Point Mass Nav, 3D Double Integrator, Bang-Bang Control, Bicycle Tracking, Pendulum Swing-up, Acrobot Swing-up, Multi-Goal Nav, Planar Reacher — via-point 参数化让所有算法统一对比
- **通用 Benchmark:** BBOB/COCO (ES), Gymnasium Classic Control (RL), VP-STO (轨迹优化)
- **可视化:** 搜索过程动画（粒子/椭圆/轨迹束在等高线上的演化）

## Implementation Phases
1. **Phase 1: Core Framework** — 项目结构 + 接口 + 随机工具 + L1 函数 ✅
2. **Phase 2: Algorithms** — CMA-ES → xNES → SVGD → MPPI + L1 验证 ✅
3. **Phase 3: Benchmark** — L2 函数 + benchmark runner + 评测指标 + 可视化
4. **Phase 4: Applications** — L3 场景（含新增 Pendulum/Acrobot/Multi-Goal/Reacher）+ via-point 适配器 + 全算法对比
5. **Phase 5: V2 Extensions** — REINFORCE (C++) + Flow Matching (PyTorch) + 跨方法对比

## Reference Papers
PDFs in `随机全局优化/` and `随机全局优化planning应用/`:
- NES/xNES: Exponential Natural Evolution Strategies, Natural Evolution Strategies
- CMA-ES/IGO: IGO paper, Information-Geometric Optimization Tutorial, Design Principles for MA-ES
- MPPI: TOR MPPI, MPPI from theory to parallel control
- SVGD: Stein Variational Gradient Descent
- Planning: Cross-entropy motion planning, VP-STO, trajectory distribution control

## Implementation Log 规则
每个 Phase 实现完成后，**必须**将遇到的重要问题记录到 `docs/implementation_notes.md`，包括：
- **Bug**：错误原因、影响范围、修复方法
- **设计认知**：算法的物理/数学限制（如收敛上限、噪声基底）
- **踩坑**：参数默认值、接口行为等非显而易见的问题

## Key Design Decisions
- 统一 Problem/Solver 接口：MPPI 和 CMA-ES/xNES 共享同一 Solver::Solve(Problem)，轨迹问题通过 ControlSequenceProblem 包装
- L3 场景默认全部使用控制序列参数化，via-point 作为可选降维实验
- 多模态搜索：SVGD 排斥核天然多模态，CMA-ES/xNES 可扩展为 MoG 提升多模态能力
- 搜索过程可视化：C++ 输出每代状态 CSV/JSON → Python matplotlib 渲染动画
