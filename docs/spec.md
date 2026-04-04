# Global Optimization Solver Library — Spec Doc

## 1. Project Overview

**名称：** `global_optim` — A C++ library for stochastic global optimization

**目标：** 提供统一接口的全局优化求解器库，包含多种基于采样的随机优化算法，支持标准 benchmark 测试及简易机器人/自动驾驶规划场景演示。

**第一版算法：**
| 算法 | 类别 | 核心思想 |
|------|------|----------|
| CMA-ES | Evolution Strategy | 协方差矩阵自适应的自然梯度优化 |
| xNES | Natural Evolution Strategy | 指数参数化的自然进化策略 |
| MPPI | Path Integral Control | 基于重要性采样的模型预测路径积分控制 |
| SVGD | Particle-based Variational Inference | 基于 Stein 算子的粒子梯度下降 |

---

## 2. Architecture

### 2.1 目录结构

```
global_optim/
├── CMakeLists.txt
├── include/
│   └── global_optim/
│       ├── core/
│       │   ├── solver.h              // Solver 统一基类
│       │   ├── problem.h             // Problem 统一接口
│       │   ├── result.h              // 求解结果
│       │   ├── types.h               // 公共类型定义
│       │   └── random.h              // 随机数工具
│       ├── solvers/
│       │   ├── cmaes.h
│       │   ├── xnes.h
│       │   ├── mppi.h
│       │   └── svgd.h
│       ├── problems/
│       │   ├── benchmark_functions.h  // Rosenbrock, Rastrigin, Ackley 等
│       │   ├── control_sequence_problem.h  // 控制序列参数化适配器
│       │   └── via_point_problem.h         // Via-point 参数化适配器（可选降维）
│       └── utils/
│           ├── logging.h
│           └── timer.h
├── src/
│   ├── core/
│   ├── solvers/
│   ├── problems/
│   └── utils/
├── playground/
│   ├── benchmark_runner.cpp           // 标准函数 benchmark
│   ├── trajectory_demo.cpp            // 简易轨迹优化演示
│   └── visualize.py                   // Python 可视化脚本
├── tests/
│   ├── test_cmaes.cpp
│   ├── test_xnes.cpp
│   ├── test_mppi.cpp
│   └── test_svgd.cpp
└── third_party/
    └── eigen/                         // Eigen 线性代数库
```

### 2.2 依赖

| 依赖 | 用途 | 必选 |
|------|------|------|
| Eigen 3.4+ | 矩阵运算 | 是 |
| Google Test | 单元测试 | 是 |
| matplotlib-cpp 或 Python | 可视化 | 可选 |

### 2.3 构建系统

CMake (>=3.16)。后续可考虑 Bazel 集成。

---

## 3. Core Interfaces

### 3.1 Problem

```cpp
namespace global_optim {

// 统一优化问题接口
// 所有问题（静态优化、轨迹优化）最终都通过此接口求解：x -> cost
class Problem {
 public:
  virtual ~Problem() = default;

  // 目标函数：x -> cost（最小化）
  virtual double Evaluate(const Eigen::VectorXd& x) const = 0;

  // 维度
  virtual int Dimension() const = 0;

  // 搜索空间边界（可选）
  virtual Eigen::VectorXd LowerBound() const { return {}; }
  virtual Eigen::VectorXd UpperBound() const { return {}; }

  // 梯度（可选，SVGD 需要）
  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& x) const {
    throw std::runtime_error("Gradient not implemented");
  }

  // 是否提供梯度
  virtual bool HasGradient() const { return false; }
};

// 轨迹优化的动力学/代价描述（不是求解接口，而是问题描述）
// 通过 Adapter 包装成 Problem 后，所有 Solver 都能求解
struct DynamicsModel {
  int state_dim;
  int control_dim;
  int horizon;          // T，时域步数
  double dt;            // 时间步长

  // 动力学模型：(state, control, dt) -> next_state
  std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                 const Eigen::VectorXd&, double)> dynamics;

  // 运行代价：(state, control) -> cost
  std::function<double(const Eigen::VectorXd&,
                        const Eigen::VectorXd&)> running_cost;

  // 终端代价：state -> cost
  std::function<double(const Eigen::VectorXd&)> terminal_cost;
};

// 控制序列参数化：x = [u_0, u_1, ..., u_{T-1}] ∈ R^{T * control_dim}
// 将轨迹优化问题包装为 Problem，所有算法统一使用
class ControlSequenceProblem : public Problem {
 public:
  ControlSequenceProblem(DynamicsModel model, Eigen::VectorXd initial_state);

  double Evaluate(const Eigen::VectorXd& x) const override {
    // x = 控制序列拼接向量
    // 1. reshape 为 [u_0, ..., u_{T-1}]
    // 2. 逐步 rollout: x_{t+1} = dynamics(x_t, u_t, dt)
    // 3. 累加 running_cost + terminal_cost
  }

  int Dimension() const override { return model_.horizon * model_.control_dim; }

 private:
  DynamicsModel model_;
  Eigen::VectorXd initial_state_;
};

// Via-point 参数化（可选降维）：x = [p_1, ..., p_N] ∈ R^{N * state_dim}
// 通过样条插值构造轨迹，再反算控制序列或直接评估路径代价
class ViaPointProblem : public Problem {
 public:
  ViaPointProblem(DynamicsModel model, Eigen::VectorXd initial_state,
                  Eigen::VectorXd goal_state, int num_via_points);

  double Evaluate(const Eigen::VectorXd& x) const override {
    // x = via-points 拼接向量
    // 1. reshape 为 [p_1, ..., p_N]
    // 2. cubic spline 插值: start → p_1 → ... → p_N → goal
    // 3. 评估路径代价 (长度 + 碰撞 + 平滑性)
    //    或反算控制后用 dynamics rollout 评估
  }

  int Dimension() const override { return num_via_points_ * state_dim_; }
};

}  // namespace global_optim
```

**设计要点：** 所有轨迹优化问题最终都包装成 `Problem`。`ControlSequenceProblem` 是默认参数化（所有算法都用），`ViaPointProblem` 是可选降维（高维问题时有优势）。没有独立的 `TrajectorySolver`——MPPI/CMA-ES/xNES 全部通过统一的 `Solver::Solve(Problem)` 求解。

### 3.2 Result

```cpp
namespace global_optim {

struct SolverResult {
  Eigen::VectorXd best_x;        // 最优解（控制序列或 via-point 向量）
  double best_cost;               // 最优代价
  int num_evaluations;            // 函数评估次数
  int num_iterations;             // 迭代次数
  double elapsed_time_ms;         // 耗时

  // 收敛历史（每代最优值）
  std::vector<double> cost_history;
};

// 辅助函数：从 SolverResult 提取轨迹信息（仅 ControlSequenceProblem 使用）
// result.best_x → reshape 为控制矩阵 → rollout → 状态轨迹
TrajectoryInfo ExtractTrajectory(const SolverResult& result,
                                  const ControlSequenceProblem& problem);

struct TrajectoryInfo {
  Eigen::MatrixXd controls;     // T x control_dim
  Eigen::MatrixXd states;       // (T+1) x state_dim
};

}  // namespace global_optim
```

### 3.3 Solver Base

```cpp
namespace global_optim {

// 配置基类
struct SolverOptions {
  int max_iterations = 1000;
  int max_evaluations = 100000;
  double cost_target = -std::numeric_limits<double>::infinity();
  bool verbose = false;
  uint64_t seed = 42;
};

// 统一求解器基类 — 所有算法（CMA-ES, xNES, SVGD, MPPI, ...）
class Solver {
 public:
  virtual ~Solver() = default;
  virtual std::string Name() const = 0;
  virtual SolverResult Solve(const Problem& problem,
                              const Eigen::VectorXd& x0) = 0;
};

}  // namespace global_optim
```

**注意：** MPPI 也是 `Solver` 的子类，输入统一的 `Problem`（实际传入 `ControlSequenceProblem`）。MPPI 内部可以 `dynamic_cast` 获取 `DynamicsModel` 来利用时序结构（逐步注入噪声、time-shift warm-start），但对外接口完全统一。

---

## 4. Algorithm Specifications

### 4.1 CMA-ES

**参考：** IGO paper, Hansen tutorial

**参数：**
```cpp
struct CMAESOptions : SolverOptions {
  int lambda = -1;           // 种群大小，默认 4 + floor(3*ln(n))
  int mu = -1;               // 父代数量，默认 lambda/2
  double sigma0 = 0.5;       // 初始步长
  // 以下参数自动根据维度计算默认值
  double c_c = -1;           // 累积路径衰减
  double c_sigma = -1;       // 步长累积衰减
  double c_1 = -1;           // rank-1 学习率
  double c_mu_coeff = -1;    // rank-mu 学习率
};
```

**核心流程：**
1. 从 `N(m, σ²C)` 采样 λ 个个体
2. 评估适应度并排序
3. 加权重组更新均值 `m`
4. 更新进化路径 `p_σ`, `p_c`
5. Rank-1 + Rank-μ 更新协方差矩阵 `C`
6. CSA 更新步长 `σ`

**复杂度：** O(n²) per generation（需周期性 O(n³) 特征分解）

### 4.2 xNES

**参考：** Exponential Natural Evolution Strategies (Glasmachers et al.)

**参数：**
```cpp
struct XNESOptions : SolverOptions {
  int lambda = -1;             // 种群大小，默认 4 + floor(3*ln(n))
  double eta_mu = 1.0;         // 均值学习率
  double eta_sigma = -1;       // 步长学习率，默认 (3+ln(n))/(5*n*sqrt(n))
  double eta_B = -1;           // 形状学习率，默认同 eta_sigma
  double sigma0 = 1.0;         // 初始 sigma
};
```

**核心流程：**
1. 从 `N(μ, σ²BB^T)` 采样：`x_i = μ + σBz_i`, `z_i ~ N(0,I)`
2. 评估并排序，计算 rank-based utilities `u_i`
3. 计算自然梯度：`G_δ = Σ u_i z_i`, `G_M = Σ u_i (z_i z_i^T - I)`
4. 分解 `G_M` 为步长分量 `G_σ = tr(G_M)/d` 和形状分量 `G_B = G_M - G_σ I`
5. 更新：`μ += η_μ σ B G_δ`, `σ *= exp(η_σ/2 · G_σ)`, `B *= expm(η_B/2 · G_B)`

**复杂度：** O(n³) per generation（矩阵指数需特征分解）

### 4.3 MPPI

**参考：** Williams et al. (IT-MPC, JGCD 2017)

**参数：**
```cpp
struct MPPIOptions {
  int num_samples = 1024;        // K，采样轨迹数
  int horizon = 50;              // T，规划时域步数
  double dt = 0.02;              // 时间步长
  double lambda = 1.0;           // 逆温度
  double alpha = 0.0;            // 探索比例 (0~1)
  Eigen::MatrixXd control_noise_cov;  // Σ，控制噪声协方差
  int max_iterations = 1;        // MPC 内循环迭代次数
  uint64_t seed = 42;
};
```

**核心流程：**
1. 为 K 条轨迹采样控制噪声 `ε_t^k ~ N(0, Σ)`
2. 前向仿真：`x_{t+1} = Dynamics(x_t, u_t + ε_t^k)`
3. 累加每条轨迹代价 `S_k = Σ RunningCost(x_t) + TerminalCost(x_T)`
4. 计算重要性权重：`w_k = exp(-(S_k - min(S))/λ) / Σ exp(...)`
5. 加权更新控制序列：`u_t += Σ w_k ε_t^k`
6. 执行 `u_0`，时移 warm-start

**复杂度：** O(K × T × dynamics_cost) per iteration，天然可并行

### 4.4 SVGD

**参考：** Liu & Wang, NIPS 2016

**注意：** SVGD 本质是分布逼近方法。用于优化时，目标分布设为 `p(x) ∝ exp(-f(x)/τ)`，粒子会收敛到低代价区域。

**参数：**
```cpp
struct SVGDOptions : SolverOptions {
  int num_particles = 50;       // 粒子数
  double step_size = 0.1;       // 学习率
  double temperature = 1.0;     // τ，温度参数
  bool use_adagrad = true;      // 使用 AdaGrad 自适应学习率
  double bandwidth = -1;        // RBF 核带宽，-1 为自适应（中值启发式）
};
```

**核心流程：**
1. 初始化 n 个粒子 `{x_i}`
2. 每次迭代：
   a. 计算 score function：`∇_x log p(x_i) = -∇f(x_i)/τ`
   b. 计算 RBF 核矩阵 `k(x_i, x_j)` 及其梯度
   c. 自适应带宽：`h = med²/log(n)`
   d. 计算 SVGD 更新：`φ(x_i) = (1/n) Σ_j [k(x_j,x_i) ∇log p(x_j) + ∇_{x_j} k(x_j,x_i)]`
   e. 更新：`x_i += ε · φ(x_i)`
3. 返回代价最低的粒子作为最优解

**需要梯度：** 是。需要目标函数提供 `Gradient(x)`。

**复杂度：** O(n² × d) per iteration

### 4.5 算法谱系：从搜索分布优化的统一视角

V1 的四个算法和后续扩展的 REINFORCE / Flow Matching 可在同一框架下理解：

```
min_θ  E_{x ~ π_θ}[ f(x) ]
```

即优化搜索分布 `π_θ` 的参数，使采样的期望代价最小。

| 算法 | 搜索分布 `π_θ` | 梯度估计 | 更新方式 | 多模态能力 |
|------|---------------|---------|---------|-----------|
| CMA-ES | `N(m, σ²C)` | 自然梯度（IGO） | rank-based 加权 MLE | 单高斯：单 mode；可扩展为 MoG |
| xNES | `N(μ, σ²BB^T)` | 自然梯度（Fisher） | 指数族自然梯度 | 同上 |
| MPPI | `N(U, Σ⊗I_T)` | 重要性采样 | IS 加权均值 | 权重集中于最优轨迹；可扩展为 MoG |
| SVGD | 粒子集合（非参） | Stein operator | 粒子直接移动 | 排斥核天然维持多样性 |
| REINFORCE | 参数化策略 `π_θ(a\|s)` | log-derivative trick | SGD / 信任域 | 取决于策略参数化 |
| Flow Matching | `T_θ # π_0` (flow 变换) | flow regression loss | 神经网络训练 | 学习到的映射天然支持多模态 |

**关键联系：**
- **MPPI ≈ IS-weighted zero-order PG**：权重 `w_k ∝ exp(-S_k/λ)` 是 softmax reward 加权
- **xNES/CMA-ES = Natural PG on Gaussian family**：Fisher metric 在指数族上有解析形式，CMA-ES 额外引入 cumulative path 稳定小样本估计
- **SVGD = Functional gradient in RKHS**：不限制分布族，直接在函数空间做梯度下降
- **Flow Matching = Amortized optimization**：离线学习 noise→solution 映射，推断时零成本采样
- **所有算法统一求解接口**：轨迹优化通过 `ControlSequenceProblem` 包装成 `Problem`，所有算法共享同一个 `Solver::Solve(Problem)` 接口

**多模态搜索能力分析：**

| 方法 | 默认多模态能力 | 扩展方案 |
|------|--------------|---------|
| CMA-ES / xNES | 单高斯，单 mode 收敛 | **Mixture of Gaussians (MoG)**：维护 K 个高斯分量，每个独立做 IGO 更新，分量间通过 niche/repulsion 避免塌缩到同一 mode |
| MPPI | 单高斯噪声，权重集中 | **MoG 噪声分布**：多个控制均值 + 噪声协方差分量 |
| SVGD | 粒子排斥核天然分散 | 天然支持，无需扩展 |
| Flow Matching | 不同噪声样本 → 不同 mode | 天然支持，无需扩展 |

IGO + MoG 是 V2 实验的重要对比维度：在 Gaussian Mixture landscape 和 Multi-Goal Nav 上，单高斯 vs MoG 的 mode coverage 差异。

**实验设计含义：**
- 所有算法在 L3 场景上默认使用 `ControlSequenceProblem`（控制序列参数化），公平对比
- `ViaPointProblem` 作为可选降维策略，单独实验评估降维的收敛加速效果
- Pendulum/Acrobot 等 episodic 场景上 MPPI 和 REINFORCE 直接可比
- xNES/CMA-ES 的协方差自适应应优于 MPPI 的固定噪声协方差
- 多模态场景（Gaussian Mixture, Multi-Goal）：单高斯 vs MoG vs SVGD vs Flow Matching 四级对比
- Flow Matching 在线推断速度远超在线优化方法，但需要离线训练成本

### 4.6 REINFORCE / Policy Gradient（V2 扩展）

**参考：** Williams 1992 (REINFORCE), Schulman et al. 2017 (PPO)

**动机：** MPPI 本质是零阶 PG，加入显式 PG baseline 可以验证这一理论联系并对比样本效率。

**参数：**
```cpp
struct REINFORCEOptions {
  int num_rollouts = 64;          // 每次迭代的 rollout 数
  int horizon = 50;               // 单次 rollout 步数
  double dt = 0.02;               // 时间步长
  double learning_rate = 1e-3;    // 策略参数学习率
  double baseline_alpha = 0.9;    // baseline 指数移动平均系数
  bool use_natural_gradient = false; // 使用自然梯度（TRPO 风格）
  int max_iterations = 500;
  uint64_t seed = 42;
};
```

**核心流程：**
1. 参数化策略 `π_θ(u_t | x_t)` 为线性高斯策略：`u = Kx + ε`, `ε ~ N(0, Σ)`
2. 采样 N 条 rollout，累计每条回报 `R_k = Σ_t r(x_t, u_t)`
3. 计算梯度估计：`ĝ = (1/N) Σ_k (R_k - b) ∇_θ log π_θ(τ_k)`，`b` 为 baseline
4. 更新参数：`θ += α · ĝ`（或自然梯度：`θ += α · F^{-1} ĝ`）

**V1 中的位置：** 不在 V1 实现范围，作为 Phase 5 扩展。仅需实现线性高斯策略版本即可完成对比实验。

### 4.7 Flow Matching（V2 扩展）

**参考：** Lipman et al. 2023 (Flow Matching), VP-STO (Trajectory flow matching)

**动机：** 将优化从在线求解转为离线学习。训练 flow 模型后，推断时只需一次前向 ODE 积分即可生成候选解，适合实时规划场景。

**思路（trajectory optimization 版本）：**
1. **数据收集：** 用 CMA-ES/MPPI 等在线方法对一组问题实例求解，收集 (问题参数, 最优轨迹) 对
2. **训练：** 学习条件向量场 `v_θ(x_t, t | c)` 使 ODE `dx/dt = v_θ(x_t, t | c)` 把高斯噪声 `x_0 ~ N(0,I)` 在 `t=1` 映射到解分布
3. **推断：** 给定新问题条件 `c`，从噪声采样 → 积分 ODE → 得到候选解
4. **可选 refinement：** flow 输出作为 warm-start，再用在线方法微调

**V1 中的位置：** 不在 V1 C++ 实现范围。Phase 5 以 Python (PyTorch) 实现，作为对比基线。C++ 侧仅需输出训练数据。

**实验设计：**
- 在 L3 场景上收集 CMA-ES/MPPI 的求解数据（不同初始条件/障碍物配置）
- 训练 Flow Matching 模型
- 对比：推断质量 vs 在线求解质量，推断速度 vs 在线求解速度
- 关键指标：one-shot 成功率、refinement 后收敛速度、多解覆盖度

---

## 5. Validation & Benchmark

验证分为三层：正确性验证 → 全局优化能力验证 → 应用场景验证。

### 5.1 Layer 1: 正确性验证（Sanity Check）

用简单凸/弱非凸函数验证算法实现的正确性。所有算法都应该能在这些问题上稳定收敛。

| 函数 | 维度 | 特点 | 验证目的 |
|------|------|------|----------|
| Sphere | 2D, 10D | 凸，最简基线 | 确认基本收敛 |
| Ellipsoid | 10D, 30D | 凸，病态条件数 | 验证协方差/步长自适应 |
| Rosenbrock | 2D, 8D | 窄弯曲谷，弱非凸 | 验证非球形搜索能力 |
| Cigar / Tablet | 10D | 各向异性极端 | 验证坐标变换学习 |

**通过标准：** 多次运行 100% 收敛到已知最优解（容差 1e-6），评估次数与论文报告量级一致。

来源：xNES 论文 Table 1, CEC 2005 benchmark set

### 5.2 Layer 2: 全局优化能力验证

这是核心验证层。问题结构必须是**非凸、多模态**的，体现全局优化的价值。

#### 5.2.1 多模态标准函数

| 函数 | 维度 | 特点 | 验证目的 |
|------|------|------|----------|
| Rastrigin | 2D, 10D, 30D | 指数级局部极值（cos扰动） | 全局搜索 vs 局部陷入 |
| Ackley | 2D, 10D | 大量局部极值 + 平坦外围 | 探索能力 |
| Schwefel | 2D, 10D | 全局最优远离次优解 | 欺骗性问题 |
| Griewank-Rosenbrock | 2D, 10D | 多模态 + 弯曲谷 | 组合难度 |

来源：BBOB benchmark (f15-f24), NES 论文实验

#### 5.2.2 Double-Rosenbrock（欺骗性双漏斗）

来源：NES 论文 Section 6.4

```
f(z) = min{ Rosenbrock(-z - 10),  5 + Rosenbrock((z - 10) / 4) }
```

- 两个极值：局部最优在 (14,14)，全局最优在 (-11,-11)
- **欺骗性**：全局结构引导算法走向局部最优
- 测试不同初始方差下找到全局最优的比例
- **验证目的：** 全局 vs 局部搜索能力，对初始分布的鲁棒性

#### 5.2.3 Gaussian Mixture Landscape（多解发现）

自定义 benchmark，专门测试算法发现**多个**局部最优的能力：

```
f(x) = -log( Σ_k  w_k * N(x; μ_k, Σ_k) )
```

- 多个 Gaussian 峰构成多个局部最优，峰值接近
- 2D 版本用于可视化：3~5 个峰
- 高维版本（10D）用于定量评估
- **验证目的：** SVGD 天然输出粒子分布，应该能发现多个峰；CMA-ES/xNES 通常只找到一个；可以对比多解发现能力

来源：SVGD 论文 1D Gaussian mixture 实验 + BBOB f21 (101 Gaussian peaks)

#### 5.2.4 Random-Basin（纯随机多模态）

来源：NES 论文 Section 6.4

```
f(z) = 1 - 0.9*r(floor(z/10)) - 0.1*r(floor(z)) * Π sin²(πz_i)^{1/(200d)}
```

- 每个单位超立方体是一个局部极值的吸引域
- 局部极值均匀分布在 [0,1]，无全局结构
- 维度：2D, 4D, 8D, 16D
- **验证目的：** 在完全随机的 landscape 中，哪个算法能找到更好的局部最优

### 5.3 Layer 3: 应用场景验证

将全局优化问题嵌入具有物理意义的场景，所有算法（包括 MPPI）都可以在这些问题上竞争。

#### 5.3.1 2D Point Mass Navigation（多同伦类路径）

来源：Cross-Entropy Motion Planning 论文, VP-STO 论文

- 2D 环境 + 障碍物 → 存在多条 homotopy class 不同的路径
- 将轨迹参数化为 via-points 向量 `z ∈ R^{2N}`（N 个 via-point）
- 代价 = 路径长度 + 碰撞惩罚 + 平滑性
- **所有算法统一接口：** 静态优化 via-point 向量
- **验证目的：**
  - 能否找到最优 homotopy class 的路径（全局性）
  - 能否发现多条可行路径（多解）
  - 收敛速度对比

#### 5.3.2 Double Integrator 3D 避障

来源：Cross-Entropy Motion Planning 论文

- 3D 环境，300 个随机球体障碍物
- 状态 6D（位置+速度），控制 3D（加速度）
- 轨迹参数化为 8 个中间状态
- **验证目的：** 高维非凸问题，多 homotopy class，测试扩展性

#### 5.3.3 1D Bang-Bang 时间最优控制

来源：VP-STO 论文 Ablation

- 双积分器 `q̈ = u`，速度约束 `|q̇| < v_max`，加速度约束 `|u| < a_max`
- 最优解是 bang-bang 控制，**解析解已知** → 可精确验证
- 参数化为 via-points，维度可调
- **验证目的：** 有解析解的非凸约束问题，定量验证精度

#### 5.3.4 Bicycle Model 路径跟踪

来源：IT-MPC (MPPI) 论文

- 简化自行车模型，状态 5D (x, y, θ, v, δ)，控制 2D (加速度, 转向角速度)
- 参考线跟踪 + 障碍物避让
- 非线性动力学 → 代价 landscape 非凸
- **验证目的：** MPPI 的主场景，其他算法通过 via-point 参数化参与对比

#### 5.3.5 Inverted Pendulum Swing-up（经典非线性控制）

来源：Gymnasium Pendulum-v1, Tedrake "Underactuated Robotics"

- 状态 2D `(θ, θ̇)`，控制 1D（力矩 `τ`），力矩有界 `|τ| ≤ τ_max`
- 动力学：`θ̈ = (τ - mgl sin(θ) - bθ̇) / (ml²)`
- 目标：从下垂 `θ=π` 摆到竖直 `θ=0` 并稳定
- **非凸性：** 需要先蓄能摆动再切换到稳定控制，代价 landscape 有多个局部极值
- **解析基线：** 能量泵浦 (energy shaping) 策略可作为参考
- **验证目的：**
  - MPPI vs REINFORCE 的直接对比（同一个 episodic 控制问题）
  - 所有算法通过 via-point 参数化 `θ(t)` 统一对比
  - 可视化：相空间 `(θ, θ̇)` 上的搜索轨迹演化

#### 5.3.6 Acrobot Swing-up（欠驱动系统）

来源：Gymnasium Acrobot-v1, Spong 1995

- 双连杆，仅第二关节有驱动，状态 4D `(θ₁, θ₂, θ̇₁, θ̇₂)`，控制 1D
- 动力学：非线性耦合二阶 ODE（惯性矩阵 + 科氏力 + 重力）
- 目标：将末端摆到指定高度以上
- **难度：** 比 Pendulum 显著更难——欠驱动 + 4D 状态 + 非线性耦合
- **验证目的：**
  - 测试全局搜索在高维非凸控制问题上的扩展性
  - CMA-ES/xNES 的自适应能力 vs MPPI 固定噪声分布
  - REINFORCE 在 sparse reward 下的表现（只有到达目标才有奖励）

#### 5.3.7 Multi-Goal 2D Navigation（多目标到达）

来源：自定义 benchmark，灵感来自 VP-STO 多解实验 + Flow Matching 多模态评估

- 2D 点质量 + 若干障碍物 + **多个等价目标区域**（3~5 个）
- 每个目标区域代价相近但路径完全不同
- 代价 = 路径长度 + 碰撞惩罚 + 目标距离
- **验证目的：**
  - **多解发现**的核心 benchmark：不是找一个最优解，而是找到**所有**可行解
  - SVGD 粒子应自然分散到多个目标（repulsive kernel）
  - Flow Matching 应学到多模态解分布
  - CMA-ES/xNES 只能收敛到一个目标 → 量化多解缺陷
  - 可视化：粒子/采样在 2D 空间中的分布演化
- **指标：** 引入 **mode coverage**（发现的目标区域数 / 总目标数）和 **mode quality**（每个 mode 内最优解质量）

#### 5.3.8 Planar Reacher（2-Link Arm）

来源：MuJoCo Reacher 简化版, Gymnasium Reacher-v4

- 平面 2-link 机械臂，状态 4D `(θ₁, θ₂, θ̇₁, θ̇₂)`，控制 2D `(τ₁, τ₂)`
- 目标：末端到达指定位置
- **多解性：** 逆运动学天然有 elbow-up / elbow-down 两组解
- 加入障碍物后：不同的 homotopy class 对应不同关节角轨迹
- **验证目的：**
  - 连续控制标准 benchmark，与 RL 文献直接可比
  - 多解性适中（2~4 个 mode），可定量评估 mode coverage
  - MPPI、REINFORCE、CMA-ES 在同一问题上的样本效率对比

### 5.4 评测指标

| 指标 | 适用层 | 说明 |
|------|--------|------|
| **收敛精度** | L1, L2 | 最终最优值与已知最优的差距 |
| **收敛曲线** | All | best cost vs. function evaluations（每代记录） |
| **全局最优成功率** | L2, L3 | N 次运行中找到全局最优（容差内）的比例 |
| **多解发现数** | L2, L3 | 找到的不同局部最优个数（通过聚类判定） |
| **评估次数** | All | 达到目标精度的 function evaluation 总数 |
| **Wall-clock time** | All | 实际计算耗时 |
| **ECDF 曲线** | L2 | 不同精度阈值下的成功率累积分布（BBOB 标准） |

每个 benchmark 运行 **50~100 次**（不同随机种子），报告中位数和四分位距。

### 5.5 搜索过程可视化

所有 2D 问题提供以下可视化：

1. **Contour + Particles/Samples 动画**
   - 背景：目标函数等高线
   - 前景：每一代的采样点位置（CMA-ES/xNES 的种群、SVGD 的粒子、MPPI 的轨迹样本）
   - 输出：逐帧 PNG → GIF/MP4

2. **搜索分布演化**
   - CMA-ES/xNES：椭圆（均值 + 协方差 3σ 等高线）随迭代变化
   - SVGD：粒子核密度估计随迭代变化
   - MPPI：轨迹束的分布随迭代变化

3. **收敛仪表盘**
   - 左：搜索过程动画
   - 右上：cost 收敛曲线
   - 右下：分布/步长参数变化

**实现方式：** C++ 求解器输出每代状态到 CSV/JSON → Python (matplotlib) 渲染动画

### 5.6 Benchmark 与算法的对应关系

| Benchmark | CMA-ES | xNES | SVGD | MPPI | REINFORCE | Flow Match |
|-----------|--------|------|------|------|-----------|------------|
| L1: 凸函数 | ✓ | ✓ | ✓ | — | — | — |
| L2: Rastrigin/Ackley/... | ✓ | ✓ | ✓ | — | — | — |
| L2: Double-Rosenbrock | ✓ | ✓ | ✓ | — | — | — |
| L2: Gaussian Mixture | ✓ | ✓ | ✓★ | — | — | ✓★★ |
| L2: Random-Basin | ✓ | ✓ | ✓ | — | — | — |
| L3: 2D Point Mass Nav | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| L3: 3D Double Integrator | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| L3: Bang-Bang Control | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| L3: Bicycle Tracking | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| L3: Pendulum Swing-up | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| L3: Acrobot Swing-up | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| L3: Multi-Goal Nav | ✓ | ✓ | ✓★ | ✓ | ✓ | ✓★★ |
| L3: Planar Reacher | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

✓★ = 天然多模态优势（SVGD 排斥核 / Flow Matching 多模态生成）
✓MoG = IGO + Mixture of Gaussians 扩展后可参与多模态对比

**L3 统一方式：**
- **默认参数化：** 所有算法统一使用 `ControlSequenceProblem`（控制序列拼接为优化向量），公平对比
- **可选降维：** `ViaPointProblem` 作为独立实验，评估 via-point 降维对收敛速度的影响
- **多模态对比：** 单高斯 vs MoG vs SVGD vs Flow Matching 在 Gaussian Mixture / Multi-Goal 上的 mode coverage
- **Flow Matching：** 用在线方法的求解数据训练，评估 amortized inference 的质量和速度

### 5.7 通用 Benchmark 标准参考

本项目的 benchmark 设计参考了以下社区标准，便于与文献结果对比：

| 标准 | 社区 | 覆盖范围 | 本项目对应 |
|------|------|---------|-----------|
| **BBOB/COCO** | 进化计算 (ES) | 24 个无噪声函数，各维度 | L1 + L2 函数直接来源 |
| **CEC Benchmark** | 进化计算 | 年度竞赛函数集（组合、旋转） | L2 补充参考 |
| **Gymnasium Classic Control** | 强化学习 (RL) | Pendulum, CartPole, Acrobot, MountainCar | L3: Pendulum, Acrobot |
| **MuJoCo Locomotion** | 强化学习 | HalfCheetah, Hopper, Walker2d, Ant | 暂不纳入（依赖重） |
| **VP-STO Suite** | 轨迹优化 | via-point 参数化 + 多种场景 | L3: Point Mass, Bang-Bang |
| **Cross-Entropy MP** | 运动规划 | 多障碍物导航 + 多 homotopy class | L3: Point Mass, Double Integrator |

**BBOB/COCO 对齐说明：**
- L1 对应 BBOB f1 (Sphere), f2 (Ellipsoid), f8 (Rosenbrock), f11 (Cigar)
- L2 对应 BBOB f15 (Rastrigin), f23 (Schwefel), f19 (Griewank-Rosenbrock)
- 输出格式兼容 COCO post-processing（ECDF 曲线、runtime distribution）

**Gymnasium 对齐说明：**
- Pendulum/Acrobot 使用相同动力学参数，使 MPPI/CMA-ES 结果可与 PPO/SAC 文献直接对比
- 区别：本项目是 open-loop 轨迹优化（固定初态，优化整条控制序列），RL 文献是 closed-loop policy learning

---

## 6. Implementation Plan

### Phase 1: Core Framework
- [ ] 项目结构搭建 + CMake 配置
- [ ] `Problem`, `Solver`, `SolverResult` 统一接口
- [ ] `DynamicsModel` + `ControlSequenceProblem` + `ViaPointProblem` 适配器
- [ ] 随机数工具
- [ ] L1 正确性验证函数实现（Sphere, Ellipsoid, Rosenbrock, Cigar）

### Phase 2: Algorithms (按顺序)
- [ ] CMA-ES 实现 + L1 正确性验证
- [ ] xNES 实现 + L1 正确性验证
- [ ] SVGD 实现 + L1 正确性验证
- [ ] MPPI 实现 + 简单动力学系统验证

### Phase 3: Benchmark & Visualization
- [ ] L2 多模态函数实现（Rastrigin, Ackley, Double-Rosenbrock, Gaussian Mixture, Random-Basin）
- [ ] Benchmark runner：所有算法 × 所有 L2 函数
- [ ] 评测指标计算框架（成功率、ECDF、多解发现）
- [ ] 结果输出 CSV + Python 可视化脚本
- [ ] 2D 搜索过程动画渲染

### Phase 4: Application Scenarios
- [ ] 2D Point Mass Navigation（多 homotopy class）
- [ ] 3D Double Integrator 避障
- [ ] 1D Bang-Bang 时间最优（解析解验证）
- [ ] Bicycle Model 路径跟踪
- [ ] Inverted Pendulum Swing-up
- [ ] Acrobot Swing-up
- [ ] Multi-Goal 2D Navigation
- [ ] Planar Reacher (2-Link Arm)
- [ ] Via-point 参数化适配器（让静态求解器求解轨迹问题）
- [ ] 所有算法 × 所有场景 对比实验

### Phase 5: Algorithm Extensions (V2)
- [ ] REINFORCE (线性高斯策略) C++ 实现
- [ ] REINFORCE 在 Pendulum / Acrobot / Point Mass Nav 上验证
- [ ] MPPI vs REINFORCE 对比实验（同一问题、同采样预算）
- [ ] Flow Matching Python 实现（PyTorch, conditional flow matching）
- [ ] L3 场景训练数据收集管线（C++ solver → CSV → PyTorch Dataset）
- [ ] Flow Matching 训练 + 推断 + 与在线方法对比
- [ ] Multi-Goal / Gaussian Mixture 上的 mode coverage 对比实验

---

## 7. Design Decisions

| 决策 | 选择 | 理由 |
|------|------|------|
| 语言 | C++17 | 性能，贴近实际工程部署 |
| 线性代数 | Eigen | 成熟、header-only、性能好 |
| 构建 | CMake | 通用、易上手 |
| 最小化 vs 最大化 | 最小化 | 优化领域惯例 |
| Solver 与 Problem 分离 | 是 | 任意 Solver × Problem 组合 |
| 统一 Problem 接口 | 是 | 轨迹优化通过 ControlSequenceProblem/ViaPointProblem 包装成 Problem，所有算法共享 Solver::Solve |
| MPPI 内部优化 | dynamic_cast | MPPI 可从 ControlSequenceProblem 获取 DynamicsModel，利用时序结构（逐步噪声注入、warm-start），但对外接口统一 |

---

## 8. Future Extensions

### V2 (Phase 5，已规划)
- REINFORCE / Policy Gradient（线性高斯策略，C++）
- Flow Matching（PyTorch，amortized trajectory optimization）
- Multi-Goal benchmark + mode coverage 评测

### V3+ (远期)
- 并行化（OpenMP / CUDA for MPPI sampling）
- Python binding (pybind11)
- Bayesian Optimization (BO)
- Cross-Entropy Method (CEM)
- PPO / TRPO（非线性神经网络策略，需 libtorch 或 Python）
- Score-based Diffusion Planning（Diffuser 风格，与 Flow Matching 对比）
- 集成到实际规划系统（mipilot Bazel workspace）
- MuJoCo 集成（更丰富的 RL benchmark）
