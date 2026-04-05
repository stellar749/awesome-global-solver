# global_optim — C++ 随机全局优化算法库

针对轨迹规划场景的随机全局优化算法集合，提供统一接口（`Solver::Solve`），所有算法可在同一问题上无缝切换对比。

## 算法（V1）

| 算法 | 类名 | 适用场景 |
|------|------|---------|
| CMA-ES | `CMAESSolver` | 通用全局优化，自适应协方差 |
| xNES | `XNESSolver` | 通用全局优化，指数自然梯度 |
| SVGD | `SVGDSolver` | 多解发现，需梯度，基于粒子的变分推断 |
| MPPI | `MPPISolver` | 轨迹优化，重要性采样控制 |

## 依赖

| 库 | 版本 | 用途 |
|----|------|------|
| CMake | ≥ 3.16 | 构建系统 |
| C++ | 17 | 语言标准 |
| Eigen | ≥ 3.4 | 矩阵运算 |
| Google Test | v1.14（自动下载） | 单元测试 |

Eigen 需要系统预装：
```bash
# Ubuntu / Debian
sudo apt install libeigen3-dev

# macOS (Homebrew)
brew install eigen
```

## 构建

```bash
cd global_optim
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

可选 CMake 选项：

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_TESTS` | ON | 构建 GTest 测试套件 |
| `BUILD_PLAYGROUND` | ON | 构建 playground 可执行文件 |

## 运行测试

```bash
cd build
ctest --output-on-failure   # 运行全部测试

# 或单独运行某个套件
./tests/test_benchmark_l2
./tests/test_cmaes
```

当前测试套件（7 个，全部应通过）：

| 测试文件 | Phase | 内容 |
|---------|-------|------|
| `test_random` | 1 | 随机数工具（正态/均匀/种子复现） |
| `test_benchmark_l1` | 1 | L1 函数正确性（Sphere/Ellipsoid/Rosenbrock/Cigar） |
| `test_cmaes` | 2 | CMA-ES 算法验证 |
| `test_xnes` | 2 | xNES 算法验证 |
| `test_svgd` | 2 | SVGD 算法验证 |
| `test_mppi` | 2 | MPPI 算法验证 |
| `test_benchmark_l2` | 3 | L2 多模态函数 + BenchmarkRunner + 指标 |

## 快速上手

### 最小示例：用 CMA-ES 优化 Rosenbrock 函数

```cpp
#include "global_optim/problems/benchmark_functions.h"
#include "global_optim/solvers/cmaes.h"

using namespace global_optim;

int main() {
    RosenbrockProblem problem(5);     // 5 维 Rosenbrock

    CMAESOptions opts;
    opts.sigma0 = 0.5;
    opts.max_iterations = 1000;
    opts.seed = 42;

    Vector x0 = Vector::Zero(5);
    SolverResult result = CMAESSolver(opts).Solve(problem, x0);

    printf("best cost: %.6e\n", result.best_cost);    // 期望 ~0
    printf("evaluations: %d\n", result.num_evaluations);
    return 0;
}
```

### 切换算法

所有算法共享相同的 `Solver::Solve(problem, x0)` 接口：

```cpp
// CMA-ES
CMAESOptions cmaes_opts; cmaes_opts.sigma0 = 1.0;
auto r1 = CMAESSolver(cmaes_opts).Solve(problem, x0);

// xNES
XNESOptions xnes_opts; xnes_opts.sigma0 = 1.0;
auto r2 = XNESSolver(xnes_opts).Solve(problem, x0);

// SVGD（需要 problem.HasGradient() == true）
SVGDOptions svgd_opts; svgd_opts.num_particles = 50;
auto r3 = SVGDSolver(svgd_opts).Solve(problem, x0);

// MPPI
MPPIOptions mppi_opts; mppi_opts.num_samples = 512;
auto r4 = MPPISolver(mppi_opts).Solve(problem, x0);
```

### 自定义优化问题

继承 `Problem` 基类，实现 `Evaluate` 和 `Dimension`：

```cpp
class MyProblem : public Problem {
public:
    explicit MyProblem(int dim) : dim_(dim) {}

    double Evaluate(const Vector& x) const override {
        return x.squaredNorm();  // your cost function
    }

    // 可选：提供梯度（SVGD 需要）
    Vector Gradient(const Vector& x) const override { return 2.0 * x; }
    bool HasGradient() const override { return true; }

    int Dimension() const override { return dim_; }

private:
    int dim_;
};
```

### 记录收敛过程

`SolverResult` 包含逐代最优值历史，可用于绘制收敛曲线：

```cpp
auto result = CMAESSolver(opts).Solve(problem, x0);

// cost_history[i] — 第 i 代的最优代价
// eval_history[i] — 第 i 代累计评估次数（作为 X 轴，公平对比不同算法）
for (size_t i = 0; i < result.cost_history.size(); ++i) {
    printf("eval=%d  cost=%.4e\n",
           result.eval_history[i], result.cost_history[i]);
}
```

### 记录种群轨迹（用于可视化动画）

```cpp
CMAESOptions opts;
opts.record_population = true;   // 启用种群快照
opts.max_iterations = 100;

auto result = CMAESSolver(opts).Solve(problem, x0);

// result.population_history[g] 是第 g 代的 (lambda × dim) 矩阵
// 保存为 CSV 供 Python 渲染动画
BenchmarkRunner::SavePopulationCSV("cmaes", "my_problem", result, "pop.csv");
```

## Benchmark Runner

运行全量 benchmark（所有算法 × 所有 L2 函数，51 个随机种子）：

```bash
cd build
./playground/benchmark_runner
# 默认输出到 benchmark_results/

# 自定义参数
./playground/benchmark_runner --seeds 20 --output-dir my_results
```

输出文件：

| 文件 | 内容 |
|------|------|
| `benchmark_results.csv` | 每次运行的汇总（solver/problem/seed/best_cost/evals） |
| `{solver}_{problem}_conv.csv` | 收敛曲线（eval vs best_cost） |

标准输出会打印汇总表：
```
Solver     Problem                 Dim      Median        Q25        Q75   Success%  Med.Evals
------------------------------------------------------------------------------------------
cmaes      rastrigin_2d              2   1.985e-06  9.926e-07  7.941e-06    100.0%       2680
xnes       rastrigin_2d              2   9.926e-07  9.926e-07  9.926e-07    100.0%       1400
...
```

## 可视化

需要 Python 环境：

```bash
pip install matplotlib numpy pandas Pillow
```

```bash
cd build/playground   # 或包含 benchmark_results/ 的目录

# 生成所有图表（收敛曲线 + box-plot + ECDF + landscape）
python visualize.py --results-dir benchmark_results --output-dir plots

# 只看某个问题的 ECDF
python visualize.py --mode ecdf --problem rastrigin_2d

# 只生成 box-plot
python visualize.py --mode boxplot

# 搜索过程动画（需要先用 record_population=true 运行并保存 CSV）
python visualize.py --mode animation --solver cmaes --problem rastrigin_2d
```

生成的图表：

| 模式 | 输出文件 | 内容 |
|------|---------|------|
| `convergence` | `convergence_{problem}.png` | 所有算法的收敛曲线（log scale） |
| `boxplot` | `boxplot_{problem}.png` | 最终代价分布 box-plot |
| `ecdf` | `ecdf_{problem}.png` | 经验累积分布函数（BBOB 风格） |
| `landscape` | `landscape_{problem}.png` | 2D 等高线图 |
| `animation` | `animation_{solver}_{problem}.gif` | 搜索过程动画 |

## 评测指标（C++ 侧）

`BenchmarkRunner` 提供以下指标计算：

```cpp
BenchmarkRunner runner(cfg);
auto records = runner.Run("cmaes", solver_fn, "rastrigin_2d", problem);

// 汇总统计（median/Q25/Q75/success_rate/median_evaluations）
auto summary = runner.Summarize(records);
printf("success rate: %.1f%%\n", 100.0 * summary.success_rate);

// ECDF 原始数据
auto [costs, cdf] = BenchmarkRunner::ComputeECDF(records);

// 多解发现（聚类 best_x）
int modes = BenchmarkRunner::CountModes(records, /*cluster_radius=*/0.5);
printf("distinct modes found: %d\n", modes);
```

## 项目结构

```
global_optim/
├── include/global_optim/
│   ├── core/
│   │   ├── problem.h          # Problem 基类接口
│   │   ├── solver.h           # Solver 基类 + SolverOptions
│   │   ├── result.h           # SolverResult（含收敛历史 + 种群快照）
│   │   ├── random.h           # 随机数工具（RandomEngine）
│   │   └── types.h            # Vector / Matrix 类型别名
│   ├── problems/
│   │   └── benchmark_functions.h  # L1 + L2 标准测试函数
│   ├── solvers/
│   │   ├── cmaes.h            # CMA-ES
│   │   ├── xnes.h             # xNES
│   │   ├── svgd.h             # SVGD
│   │   └── mppi.h             # MPPI
│   └── benchmark/
│       └── runner.h           # BenchmarkRunner（多种子运行 + 指标）
├── tests/                     # GTest 测试套件
├── playground/
│   ├── benchmark_runner.cpp   # 完整 benchmark 可执行文件
│   └── visualize.py           # Python 可视化脚本
└── CMakeLists.txt
```

## L1 / L2 测试函数一览

### L1（正确性验证，单峰）

| 函数 | 全局最优 | 特点 |
|------|---------|------|
| `SphereProblem(dim)` | x*=0, f*=0 | 各向同性基线 |
| `EllipsoidProblem(dim)` | x*=0, f*=0 | 病态条件数（~n） |
| `RosenbrockProblem(dim)` | x*=1, f*=0 | 弯曲窄谷 |
| `CigarProblem(dim)` | x*=0, f*=0 | 极端病态（条件数 1e6） |

### L2（全局优化验证，多模态）

| 函数 | 全局最优 | 特点 |
|------|---------|------|
| `RastriginProblem(dim)` | x*=0, f*=0 | 指数级局部极值（cos 扰动），边界 [-5.12, 5.12] |
| `AckleyProblem(dim)` | x*=0, f*=0 | 大量局部极值 + 平坦外围 |
| `SchwefelProblem(dim)` | x*≈420.97, f*≈0 | 欺骗性，全局最优远离次优解 |
| `DoubleRosenbrockProblem(dim)` | z*≈-11, f*=0 | 双漏斗，局部最优在 z≈14 |
| `GaussianMixtureProblem(dim,k)` | 主 mode 处 | 多模态，SVGD 可发现多个 mode |
| `GriewankProblem(dim)` | x*=0, f*=0 | 规则网格极值，坐标积 |
| `RandomBasinProblem(dim)` | 随机盆地 | 无全局结构，纯随机 landscape，边界 [-50, 50] |
