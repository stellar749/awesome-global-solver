# global_optim — C++ 随机全局优化算法库

针对轨迹规划场景的随机全局优化算法集合，提供统一接口（`Solver::Solve`），所有算法可在同一问题上无缝切换对比。

## 算法（V1）

| 算法 | 类名 | 适用场景 |
|------|------|---------|
| CMA-ES | `CMAESSolver` | 通用全局优化，自适应协方差矩阵 |
| xNES | `XNESSolver` | 通用全局优化，指数自然梯度 |
| SVGD | `SVGDSolver` | 多解发现，需梯度，基于粒子的变分推断 |
| MPPI | `MPPISolver` | 轨迹优化，重要性采样控制 |

---

## 依赖

| 库 | 版本 | 用途 |
|----|------|------|
| CMake | ≥ 3.16 | 构建系统 |
| C++ | 17 | 语言标准 |
| Eigen | ≥ 3.4 | 矩阵运算 |
| Google Test | v1.14（自动下载） | 单元测试 |
| Python | ≥ 3.9 | 可视化脚本 |
| matplotlib / numpy / pandas | — | Python 可视化依赖 |

```bash
# 安装 Eigen（Ubuntu / Debian）
sudo apt install libeigen3-dev

# 安装 Python 可视化依赖
pip install matplotlib numpy pandas
```

---

## 构建

```bash
cd global_optim
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

| CMake 选项 | 默认 | 说明 |
|-----------|------|------|
| `BUILD_TESTS` | ON | 构建 GTest 测试套件 |
| `BUILD_PLAYGROUND` | ON | 构建 benchmark_runner 可执行文件 |

---

## 快速上手（run.py 一键脚本）

所有日常操作通过 `playground/run.py` 完成，**在 `build/` 目录下运行**：

```bash
cd build

# 搜索过程动画 — 生成交互式 HTML，浏览器打开，底部滑块逐帧查看
python ../playground/run.py animate cmaes rastrigin_2d
python ../playground/run.py animate svgd  gauss_mix_2d
python ../playground/run.py animate xnes  dbl_rosen_2d
python ../playground/run.py animate mppi  ackley_2d

# 多算法对比仪表板 — 收敛曲线 + ECDF + box-plot + 汇总表
python ../playground/run.py compare rastrigin_2d
python ../playground/run.py compare gauss_mix_2d --solvers cmaes,xnes,svgd --seeds 30

# 完整 benchmark（所有算法 × 所有 L2 函数）
python ../playground/run.py benchmark --seeds 51
```

结果保存在 `results/plots/`，`.html` 文件用浏览器打开。

### run.py 参数说明

```
animate <solver> <problem> [--output-dir DIR]
compare <problem> [--solvers S1,S2,...] [--seeds N] [--output-dir DIR]
benchmark [--seeds N] [--output-dir DIR]
```

**可用 solver：** `cmaes` / `xnes` / `svgd` / `mppi`

**可用 problem（2D 支持动画）：**

| 名称 | 函数 |
|------|------|
| `rastrigin_2d` / `rastrigin_10d` | Rastrigin |
| `ackley_2d` / `ackley_10d` | Ackley |
| `schwefel_2d` | Schwefel |
| `dbl_rosen_2d` | Double-Rosenbrock（双漏斗） |
| `gauss_mix_2d` | Gaussian Mixture（多模态） |
| `griewank_2d` / `griewank_10d` | Griewank |
| `random_basin_2d` / `random_basin_4d` | Random-Basin |

---

## 可视化说明

### animate 模式

生成 `animation_{solver}_{problem}.html`，包含：
- **左图**：函数等高线 + 每代粒子位置，白星为当代最优（MPPI 额外显示红色 X 质心）
- **右图**：实时收敛曲线
- **底部滑块**：拖动逐帧查看，或点击播放

坐标范围自动根据粒子实际位置调整，不会出现粒子跑出画面的情况。

### compare 模式

生成 `compare_{problem}.html`，4 个面板：

| 面板 | 内容 | 用途 |
|------|------|------|
| 左上：收敛曲线 | 中位数 ± IQR 色带（多种子统计） | 收敛速度与稳定性 |
| 右上：ECDF | 达到某精度阈值的成功率累积分布 | 算法可靠性对比 |
| 左下：Box plot | 最终代价分布（log 坐标） | 最终结果质量 |
| 右下：汇总表 | Median / Q25/Q75 / Success% / Evals | 关键指标一览 |

---

## 运行测试

```bash
cd build
ctest --output-on-failure     # 运行全部 7 个测试套件
```

| 测试文件 | Phase | 内容 |
|---------|-------|------|
| `test_random` | 1 | 随机数工具 |
| `test_benchmark_l1` | 1 | L1 函数正确性（Sphere/Ellipsoid/Rosenbrock/Cigar） |
| `test_cmaes` | 2 | CMA-ES 算法验证 |
| `test_xnes` | 2 | xNES 算法验证 |
| `test_svgd` | 2 | SVGD 算法验证 |
| `test_mppi` | 2 | MPPI 算法验证 |
| `test_benchmark_l2` | 3 | L2 多模态函数 + BenchmarkRunner + 评测指标 |

---

## C++ API

### 最小示例

```cpp
#include "global_optim/problems/benchmark_functions.h"
#include "global_optim/solvers/cmaes.h"

using namespace global_optim;

int main() {
    RosenbrockProblem problem(5);

    CMAESOptions opts;
    opts.sigma0         = 0.5;
    opts.max_iterations = 1000;
    opts.seed           = 42;

    SolverResult result = CMAESSolver(opts).Solve(problem, Vector::Zero(5));
    printf("best cost: %.6e  evals: %d\n",
           result.best_cost, result.num_evaluations);
}
```

### 切换算法（统一接口）

```cpp
// 所有算法都实现 Solver::Solve(problem, x0)
CMAESOptions cmaes_opts; cmaes_opts.sigma0 = 1.0;
auto r1 = CMAESSolver(cmaes_opts).Solve(problem, x0);

XNESOptions xnes_opts; xnes_opts.sigma0 = 1.0;
auto r2 = XNESSolver(xnes_opts).Solve(problem, x0);

SVGDOptions svgd_opts; svgd_opts.num_particles = 50;   // 需要 HasGradient()
auto r3 = SVGDSolver(svgd_opts).Solve(problem, x0);

MPPIOptions mppi_opts; mppi_opts.num_samples = 512;
auto r4 = MPPISolver(mppi_opts).Solve(problem, x0);
```

### 自定义问题

```cpp
class MyProblem : public Problem {
public:
    explicit MyProblem(int dim) : dim_(dim) {}

    double Evaluate(const Vector& x) const override {
        return x.squaredNorm();
    }
    // 可选梯度（SVGD 需要）
    Vector Gradient(const Vector& x) const override { return 2.0 * x; }
    bool HasGradient() const override { return true; }

    int Dimension() const override { return dim_; }
private:
    int dim_;
};
```

### SolverResult 字段

```cpp
SolverResult result = solver.Solve(problem, x0);

result.best_x            // 最优解向量
result.best_cost         // 最优代价值
result.num_evaluations   // 总函数评估次数
result.num_iterations    // 总迭代代数
result.elapsed_time_ms   // 耗时（毫秒）

// 收敛历史（每代一个点，eval_history 作为 X 轴更公平）
result.cost_history      // vector<double>：每代最优代价
result.eval_history      // vector<int>：每代累计评估次数

// 种群快照（需要 opts.record_population = true）
result.population_history       // vector<Matrix>：每代 (lambda×dim) 样本位置
result.population_eval_history  // vector<int>：对应的累计评估次数
```

### BenchmarkRunner（C++ 侧评测）

```cpp
BenchmarkRunner::Config cfg;
cfg.num_seeds         = 51;
cfg.success_threshold = 1e-3;
BenchmarkRunner runner(cfg);

// 运行多种子，收集 RunRecord
SolverFn fn = [](const Problem& p, const Vector& x0, uint64_t seed) {
    CMAESOptions opts; opts.seed = seed;
    return CMAESSolver(opts).Solve(p, x0);
};
auto records = runner.Run("cmaes", fn, "rastrigin_2d", problem);

// 汇总统计
auto summary = runner.Summarize(records);
printf("median=%.3e  success=%.1f%%\n",
       summary.median_cost, 100.0 * summary.success_rate);

// ECDF 原始数据
auto [costs, cdf] = BenchmarkRunner::ComputeECDF(records);

// 多解发现（对 best_x 做贪心聚类）
int modes = BenchmarkRunner::CountModes(records, /*cluster_radius=*/0.5);

// 导出 CSV
BenchmarkRunner::SaveCSV(records, "results.csv");
BenchmarkRunner::SaveConvergenceCSV(records[0], "conv.csv");
BenchmarkRunner::SavePopulationCSV("cmaes", "problem", result, "pop.csv");
```

---

## 项目结构

```
global_optim/
├── include/global_optim/
│   ├── core/
│   │   ├── problem.h              # Problem 基类（Evaluate / Gradient / Dimension）
│   │   ├── solver.h               # Solver 基类 + SolverOptions（含 record_population）
│   │   ├── result.h               # SolverResult（收敛历史 + 种群快照）
│   │   ├── random.h               # RandomEngine 工具
│   │   └── types.h                # Vector / Matrix 别名
│   ├── problems/
│   │   └── benchmark_functions.h  # L1 + L2 全部测试函数
│   ├── solvers/
│   │   ├── cmaes.h                # CMA-ES
│   │   ├── xnes.h                 # xNES
│   │   ├── svgd.h                 # SVGD
│   │   └── mppi.h                 # MPPI
│   └── benchmark/
│       └── runner.h               # BenchmarkRunner（多种子 / ECDF / CountModes / CSV 导出）
├── tests/                         # GTest 测试套件（7 个）
├── playground/
│   ├── run.py                     # 一键集成脚本（animate / compare / benchmark）
│   ├── benchmark_runner.cpp       # C++ benchmark 可执行文件
│   └── visualize.py               # Python 可视化（含 compare_dashboard）
├── CMakeLists.txt
└── README.md
```

---

## 测试函数一览

### L1（正确性验证，单峰）

| 函数 | 最优 | 特点 |
|------|------|------|
| `SphereProblem(dim)` | x\*=0, f\*=0 | 各向同性基线 |
| `EllipsoidProblem(dim)` | x\*=0, f\*=0 | 病态条件数（~n） |
| `RosenbrockProblem(dim)` | x\*=1, f\*=0 | 弯曲窄谷 |
| `CigarProblem(dim)` | x\*=0, f\*=0 | 极端病态（条件数 1e6） |

### L2（全局优化验证，多模态）

| 函数 | 最优 | 特点 |
|------|------|------|
| `RastriginProblem(dim)` | x\*=0, f\*=0 | 指数级局部极值，边界 ±5.12 |
| `AckleyProblem(dim)` | x\*=0, f\*=0 | 平坦外围 + 局部极值密集 |
| `SchwefelProblem(dim)` | x\*≈420.97, f\*≈0 | 欺骗性，全局最优远离次优解 |
| `DoubleRosenbrockProblem(dim)` | z\*≈−11, f\*=0 | 双漏斗，初始梯度指向局部极值 |
| `GaussianMixtureProblem(dim,k)` | 主 mode | 多模态，SVGD 可发现多个 mode |
| `GriewankProblem(dim)` | x\*=0, f\*=0 | 规则网格极值，坐标乘积耦合 |
| `RandomBasinProblem(dim)` | 随机盆地 | 无全局结构，纯随机 landscape，边界 ±50 |
