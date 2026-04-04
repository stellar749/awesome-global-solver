# Implementation Notes — 重要问题与经验记录

每个 Phase 完成后记录遇到的 bug、设计决策和踩坑，供后续开发参考。

---

## Phase 2: Algorithms（CMA-ES / xNES / MPPI / SVGD）

### Bug 1: xNES 学习率公式缺少 `n` 因子

**问题**  
xNES 的步长学习率 `eta_sigma` 写成了：
```cpp
(3.0 + std::log(n)) / (5.0 * std::sqrt(n))   // ❌ 错误
```
正确公式（来自 Glasmachers et al. 2010, Table 1）：
```cpp
(3.0 + std::log(n)) / (5.0 * n * std::sqrt(n))  // ✓ 正确
```

**影响**  
漏掉 `n` 导致 n=10 时学习率大 10 倍（0.335 vs 0.0335），sigma 每代急速膨胀，算法发散而非收敛。4 个 L1 验证测试全部失败。

**教训**  
实现 NES 类算法时，学习率 `η` 通常随维度 `n` 以 `n^{3/2}` 或 `n^2` 衰减，不应只除 `sqrt(n)`。每个默认超参都要对照原论文 Table/Appendix 逐字核对。

---

### Bug 2: SVGD 粒子初始化 spread 过小

**问题**  
粒子初始化时使用固定 spread=0.1：
```cpp
X.row(i) = x0 + 0.1 * rng.RandNVector(d);  // ❌ 太小
```

**影响**  
所有粒子集中在半径 0.1 的小球内，中值带宽 `h = med² / log(n)` 接近 0。  
- `h → 0` 导致核梯度项 `(2/h) * Δx * k_ij` 趋于发散，repulsive force 极大
- 同时所有粒子 score 几乎相同，IS 权重退化为均匀分布
- 两个效果叠加：粒子既无法向 mode 收敛，又被 repulsive force 推散

**修复**  
将 spread 改为可配置参数 `init_std`（默认 1.0），测试中让粒子有合理的初始带宽。

---

### 设计认知 1: SVGD 的收敛精度有物理上限

**背景**  
SVGD 本质是逼近 Boltzmann 分布 `p(x) ∝ exp(-f(x)/τ)` 的粒子方法，不是点收敛的优化器。RBF 核的 repulsive 项会阻止粒子全部塌缩到同一点。

**数学推导**  
在 n 个粒子的均衡态，粒子保持有限 spread δ。对 Sphere 问题：
- 均衡带宽：`h_eq ≈ d * δ² / log(n)`
- score 与 repulsive 平衡：`2x_eq/τ ≈ (2/h_eq) * δ * k_avg`
- 均衡代价（最优粒子）≈ `O(τ * log(n) / (n * d))`

对 τ=0.01, n=50, d=5：最优粒子代价理论极限约 `8e-4`，而非 0。

**对 L1 测试的影响**  
SVGD 的收敛测试不应使用绝对阈值（如 1e-8），而应：
1. 用相对改善量（如 "cost < initial * 1%"）
2. 或用符合均衡态估计的松弛阈值

**AdaGrad 的额外问题**  
SVGD 配合 AdaGrad 时，前期大梯度（小 τ → 大 score）会使历史梯度平方累积量极大，永久性压低后期步长（"AdaGrad 学习率死亡"）。用于纯优化时推荐关闭 AdaGrad（`use_adagrad=false`）或改用 RMSProp。

---

### 设计认知 2: MPPI 的 IS 噪声基底

**背景**  
MPPI 的每步更新 `x ← IS_mean(x + ε)` 在 x 接近最优点后，更新量退化为有限采样的噪声均值。

**数学推导**  
在 x ≈ 0 时，K 个扰动 `ε_k ~ N(0, σ²I)` 的 IS 加权均值方差约为：
```
Var[IS_mean(ε)] ≈ σ² · d / K   (per component, 近似)
```
对 σ=0.3, K=512, d=5：噪声基底 ≈ `0.09 * 5 / 512 ≈ 8.8e-4`

这是无法通过增加迭代次数消除的极限，只能通过增大 K 或减小 σ 来降低。

**对测试设计的影响**  
MPPI 的 L1 收敛阈值应留余量：对 5D Sphere 用 `5e-4`（不是 `1e-4`）。

---

### Bug 3: MPPI `max_evaluations` 提前截断迭代

**问题**  
`SolverOptions::max_evaluations = 100000`（默认值），而 MPPI 每次迭代消耗 `K + 1` 次评估。当 `K=512`, `max_iterations=400` 时：
- 实际最多运行 `100000 / 513 ≈ 195` 次迭代，而非设定的 400 次
- 表现为增加 `max_iterations` 没有任何效果

**修复**  
对高 K 的 MPPI 测试，需要同步调高 `max_evaluations`：
```cpp
opts.max_evaluations = K * max_iterations * 2;  // 给足余量
```

---

## Phase 1: Core Framework

无重大问题。仅有一个 `-Wunused-parameter` warning（`Problem::Gradient` 基类默认实现的参数名），用 `/*x*/` 注释掉即可。
