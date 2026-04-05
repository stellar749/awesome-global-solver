#include "global_optim/problems/benchmark_functions.h"
#include "global_optim/benchmark/runner.h"
#include "global_optim/solvers/cmaes.h"
#include "global_optim/solvers/xnes.h"
#include "global_optim/solvers/svgd.h"

#include <gtest/gtest.h>
#include <cmath>

using namespace global_optim;

// ── Rastrigin ─────────────────────────────────────────────────────────────────

TEST(Rastrigin, GlobalMinimumIsZero) {
  RastriginProblem p(5);
  EXPECT_NEAR(p.Evaluate(Vector::Zero(5)), 0.0, 1e-12);
}

TEST(Rastrigin, KnownValue_1D) {
  // f([1]) = 10 + 1 - 10*cos(2π) = 10 + 1 - 10 = 1
  RastriginProblem p(1);
  Vector x(1); x << 1.0;
  EXPECT_NEAR(p.Evaluate(x), 1.0, 1e-12);
}

TEST(Rastrigin, HasManyLocalMinima) {
  // Every integer vector is a local minimum with f = 10*n
  RastriginProblem p(2);
  Vector x1(2); x1 << 1.0, 0.0;
  Vector x2(2); x2 << 0.0, 1.0;
  // These are local minima (cos(2π)=1, so grad=0) with f=10 per non-zero dim
  EXPECT_NEAR(p.Evaluate(x1), 1.0, 1e-10);
  EXPECT_NEAR(p.Evaluate(x2), 1.0, 1e-10);
}

TEST(Rastrigin, GradientAtOrigin) {
  RastriginProblem p(4);
  EXPECT_TRUE(p.Gradient(Vector::Zero(4)).isZero(1e-12));
}

TEST(Rastrigin, GradientKnownValue) {
  RastriginProblem p(2);
  Vector x(2); x << 0.5, 0.0;
  // grad[0] = 2*0.5 + 20π*sin(π) ≈ 1.0 + 0 = 1.0
  // grad[1] = 0
  Vector g = p.Gradient(x);
  EXPECT_NEAR(g[0], 1.0, 1e-10);
  EXPECT_NEAR(g[1], 0.0, 1e-12);
}

TEST(Rastrigin, Bounds) {
  RastriginProblem p(3);
  EXPECT_NEAR(p.LowerBound()[0], -5.12, 1e-10);
  EXPECT_NEAR(p.UpperBound()[0],  5.12, 1e-10);
}

// ── Ackley ────────────────────────────────────────────────────────────────────

TEST(Ackley, GlobalMinimumIsZero) {
  AckleyProblem p(5);
  EXPECT_NEAR(p.Evaluate(Vector::Zero(5)), 0.0, 1e-10);
}

TEST(Ackley, PositiveEverywhere) {
  AckleyProblem p(3);
  // Ackley is non-negative with global min 0
  for (double x : {-5.0, -1.0, 0.5, 2.0, 10.0}) {
    Vector v = Vector::Constant(3, x);
    EXPECT_GE(p.Evaluate(v), -1e-10);
  }
}

TEST(Ackley, Dimension) {
  EXPECT_EQ(AckleyProblem(7).Dimension(), 7);
}

// ── Schwefel ──────────────────────────────────────────────────────────────────

TEST(Schwefel, GlobalMinimumNearZero) {
  SchwefelProblem p(2);
  Vector x = Vector::Constant(2, 420.9687);
  EXPECT_LT(p.Evaluate(x), 1e-3);  // known global min ≈ 0
}

TEST(Schwefel, Bounds) {
  SchwefelProblem p(3);
  EXPECT_NEAR(p.LowerBound()[0], -500.0, 1e-10);
}

// ── DoubleRosenbrock ──────────────────────────────────────────────────────────

TEST(DoubleRosenbrock, GlobalMinimumNearMinus11) {
  DoubleRosenbrockProblem p(2);
  Vector x_global(2); x_global << -11.0, -11.0;
  EXPECT_NEAR(p.Evaluate(x_global), 0.0, 1e-6);
}

TEST(DoubleRosenbrock, LocalMinimumNear14) {
  DoubleRosenbrockProblem p(2);
  Vector x_local(2); x_local << 14.0, 14.0;
  EXPECT_NEAR(p.Evaluate(x_local), 5.0, 1e-6);
}

TEST(DoubleRosenbrock, GlobalBetterThanLocal) {
  DoubleRosenbrockProblem p(2);
  Vector x_global(2); x_global << -11.0, -11.0;
  Vector x_local(2);  x_local  << 14.0,  14.0;
  EXPECT_LT(p.Evaluate(x_global), p.Evaluate(x_local));
}

TEST(DoubleRosenbrock, RequiresDim2) {
  EXPECT_THROW(DoubleRosenbrockProblem(1), std::invalid_argument);
}

// ── GaussianMixture ───────────────────────────────────────────────────────────

TEST(GaussianMixture, ModeIsMinimum) {
  // For 1-mode GMM, minimum is at the mode
  GaussianMixtureProblem::Mode m;
  m.mean  = Vector::Zero(2);
  m.sigma = 1.0;
  m.weight = 1.0;
  std::vector<GaussianMixtureProblem::Mode> modes = {m};
  GaussianMixtureProblem p(modes);

  // At mode center: f = -log(N(0; 0, I)) = 0.5*d*log(2π) (normalizer only)
  double f_at_mode = p.Evaluate(Vector::Zero(2));
  double f_away    = p.Evaluate(Vector::Constant(2, 3.0));
  EXPECT_LT(f_at_mode, f_away);
}

TEST(GaussianMixture, GradientAtMode) {
  // At the mode of a unimodal GMM, gradient ≈ 0
  GaussianMixtureProblem::Mode m;
  m.mean = Vector::Zero(3); m.sigma = 1.0; m.weight = 1.0;
  std::vector<GaussianMixtureProblem::Mode> modes = {m};
  GaussianMixtureProblem p(modes);
  Vector g = p.Gradient(Vector::Zero(3));
  EXPECT_LT(g.norm(), 1e-10);
}

TEST(GaussianMixture, MultipleModesExist) {
  // 3-mode 2D GMM: three distinct local minima
  GaussianMixtureProblem p(2, 3, 4.0);
  EXPECT_EQ(static_cast<int>(p.Modes().size()), 3);
}

TEST(GaussianMixture, HasGradient) {
  EXPECT_TRUE(GaussianMixtureProblem(2, 3).HasGradient());
}

// ── Griewank ──────────────────────────────────────────────────────────────────

TEST(Griewank, GlobalMinimumIsOne_At_Origin) {
  // f(0) = 1 + 0 - Π cos(0) = 1 + 0 - 1 = 0
  GriewankProblem p(5);
  EXPECT_NEAR(p.Evaluate(Vector::Zero(5)), 0.0, 1e-12);
}

TEST(Griewank, GradientAtOrigin) {
  GriewankProblem p(4);
  Vector g = p.Gradient(Vector::Zero(4));
  EXPECT_TRUE(g.isZero(1e-12));
}

TEST(Griewank, HasGradient) {
  EXPECT_TRUE(GriewankProblem(3).HasGradient());
}

// ── BenchmarkRunner ───────────────────────────────────────────────────────────

TEST(BenchmarkRunner, RunReturnsCorrectCount) {
  BenchmarkRunner::Config cfg;
  cfg.num_seeds = 5;
  BenchmarkRunner runner(cfg);

  RastriginProblem p(2);
  SolverFn fn = [](const Problem& prob, const Vector& x0, uint64_t seed) {
    CMAESOptions opts; opts.seed = seed; opts.max_iterations = 50;
    return CMAESSolver(opts).Solve(prob, x0);
  };

  auto records = runner.Run("cmaes", fn, "rastrigin_2d", p);
  EXPECT_EQ(static_cast<int>(records.size()), 5);
  for (const auto& r : records) {
    EXPECT_EQ(r.solver_name, "cmaes");
    EXPECT_EQ(r.problem_name, "rastrigin_2d");
    EXPECT_GT(r.num_evaluations, 0);
  }
}

TEST(BenchmarkRunner, SummarizeComputesStats) {
  BenchmarkRunner::Config cfg;
  cfg.num_seeds = 11;
  cfg.success_threshold = 1.0;
  BenchmarkRunner runner(cfg);

  SphereProblem p(3);
  SolverFn fn = [](const Problem& prob, const Vector& x0, uint64_t seed) {
    CMAESOptions opts; opts.seed = seed; opts.max_iterations = 200;
    return CMAESSolver(opts).Solve(prob, x0);
  };

  auto records = runner.Run("cmaes", fn, "sphere_3d", p,
    [](uint64_t) { return Vector::Ones(3) * 2.0; });
  auto summary = runner.Summarize(records);

  EXPECT_EQ(summary.n_runs, 11);
  EXPECT_GE(summary.success_rate, 0.0);
  EXPECT_LE(summary.success_rate, 1.0);
  EXPECT_LE(summary.q25_cost, summary.median_cost);
  EXPECT_LE(summary.median_cost, summary.q75_cost);
}

TEST(BenchmarkRunner, EvalHistoryPopulated) {
  BenchmarkRunner::Config cfg;
  cfg.num_seeds = 1;
  BenchmarkRunner runner(cfg);

  SphereProblem p(2);
  SolverFn fn = [](const Problem& prob, const Vector& x0, uint64_t seed) {
    CMAESOptions opts; opts.seed = seed; opts.max_iterations = 50;
    return CMAESSolver(opts).Solve(prob, x0);
  };

  auto records = runner.Run("cmaes", fn, "sphere_2d", p);
  ASSERT_FALSE(records.empty());
  EXPECT_EQ(records[0].cost_history.size(), records[0].eval_history.size());
  EXPECT_FALSE(records[0].eval_history.empty());
}

// ── RandomBasin ───────────────────────────────────────────────────────────────

TEST(RandomBasin, BoundsCorrect) {
  RandomBasinProblem p(3);
  EXPECT_NEAR(p.LowerBound()[0], -50.0, 1e-10);
  EXPECT_NEAR(p.UpperBound()[0],  50.0, 1e-10);
}

TEST(RandomBasin, ValuesInRange) {
  // f(z) ∈ [0, 1] approximately (sin-product ≤ 1, R ∈ [0,1])
  RandomBasinProblem p(2);
  for (double v : {0.5, 1.3, -2.7, 10.1, -23.4}) {
    Vector z(2); z << v, v;
    double f = p.Evaluate(z);
    EXPECT_GE(f, -0.1);   // numerical tolerance
    EXPECT_LE(f,  1.1);
  }
}

TEST(RandomBasin, DimensionCorrect) {
  EXPECT_EQ(RandomBasinProblem(4).Dimension(), 4);
}

TEST(RandomBasin, NoGradient) {
  EXPECT_FALSE(RandomBasinProblem(2).HasGradient());
}

TEST(RandomBasin, DifferentBasinsHaveDifferentValues) {
  // Two points in different unit basins should (almost always) differ
  RandomBasinProblem p(2);
  Vector z1(2); z1 << 0.5, 0.5;   // basin (0,0)
  Vector z2(2); z2 << 1.5, 1.5;   // basin (1,1)
  // Very unlikely to be exactly equal
  EXPECT_NE(p.Evaluate(z1), p.Evaluate(z2));
}

// ── ECDF and mode coverage metrics ───────────────────────────────────────────

TEST(BenchmarkMetrics, ECDFMonotone) {
  BenchmarkRunner::Config cfg;
  cfg.num_seeds = 10;
  BenchmarkRunner runner(cfg);
  SphereProblem p(2);
  SolverFn fn = [](const Problem& prob, const Vector& x0, uint64_t seed) {
    CMAESOptions opts; opts.seed = seed; opts.max_iterations = 100;
    return CMAESSolver(opts).Solve(prob, x0);
  };
  auto records = runner.Run("cmaes", fn, "sphere_2d", p);
  auto [costs, cdf] = BenchmarkRunner::ComputeECDF(records);
  ASSERT_EQ(costs.size(), cdf.size());
  for (size_t i = 1; i < costs.size(); ++i) {
    EXPECT_LE(costs[i - 1], costs[i]);
    EXPECT_LE(cdf[i - 1],   cdf[i]);
  }
  EXPECT_NEAR(cdf.back(), 1.0, 1e-10);
}

TEST(BenchmarkMetrics, CountModesTwoBasins) {
  // Build two records with best_x far apart → should find 2 modes
  RunRecord r1, r2, r3;
  r1.best_x = Vector::Zero(2);
  r2.best_x = Vector::Constant(2, 10.0);  // far away
  r3.best_x = Vector::Constant(2, 0.1);   // close to r1
  auto modes = BenchmarkRunner::CountModes({r1, r2, r3}, 1.0);
  EXPECT_EQ(modes, 2);
}

TEST(BenchmarkMetrics, PopulationHistoryRecorded) {
  SphereProblem p(2);
  CMAESOptions opts;
  opts.max_iterations = 10;
  opts.record_population = true;
  auto result = CMAESSolver(opts).Solve(p, Vector::Zero(2));
  EXPECT_EQ(result.population_history.size(), result.cost_history.size());
  EXPECT_FALSE(result.population_history.empty());
  // Each snapshot is (lambda x 2)
  for (const auto& snap : result.population_history) {
    EXPECT_EQ(snap.cols(), 2);
    EXPECT_GT(snap.rows(), 0);
  }
}

// ── Algorithm convergence on selected L2 functions ────────────────────────────

TEST(L2Validation, CMAESFindsRastrigin2DGlobalOpt) {
  // Run 10 seeds; at least 50% should find f < 0.1 (near global min)
  // CMA-ES on 2D Rastrigin needs a large population to avoid local minima.
  BenchmarkRunner::Config cfg;
  cfg.num_seeds = 10;
  cfg.success_threshold = 0.1;
  BenchmarkRunner runner(cfg);

  RastriginProblem p(2);
  SolverFn fn = [](const Problem& prob, const Vector& x0, uint64_t seed) {
    CMAESOptions opts;
    opts.seed = seed;
    opts.max_iterations = 2000;
    opts.sigma0 = 2.0;
    opts.lambda = 20;   // larger pop improves global search on Rastrigin
    return CMAESSolver(opts).Solve(prob, x0);
  };
  auto records = runner.Run("cmaes", fn, "rastrigin_2d", p,
    [](uint64_t seed) {
      std::mt19937_64 rng(seed);
      std::uniform_real_distribution<double> u(-3.0, 3.0);
      Vector x0(2); x0 << u(rng), u(rng);
      return x0;
    });
  auto summary = runner.Summarize(records);
  EXPECT_GE(summary.success_rate, 0.5);
}

TEST(L2Validation, XNESFindsGriewank2DGlobalOpt) {
  BenchmarkRunner::Config cfg;
  cfg.num_seeds = 10;
  cfg.success_threshold = 0.1;
  BenchmarkRunner runner(cfg);

  GriewankProblem p(2);
  SolverFn fn = [](const Problem& prob, const Vector& x0, uint64_t seed) {
    XNESOptions opts;
    opts.seed = seed;
    opts.max_iterations = 500;
    opts.sigma0 = 100.0;
    return XNESSolver(opts).Solve(prob, x0);
  };
  auto records = runner.Run("xnes", fn, "griewank_2d", p,
    [](uint64_t seed) {
      std::mt19937_64 rng(seed);
      std::uniform_real_distribution<double> u(-300.0, 300.0);
      Vector x0(2); x0 << u(rng), u(rng);
      return x0;
    });
  auto summary = runner.Summarize(records);
  EXPECT_GE(summary.success_rate, 0.5);
}

TEST(L2Validation, SVGDFindsGaussianMixtureModes) {
  // SVGD with multiple particles should find a mode with low cost
  SVGDOptions opts;
  opts.max_iterations = 1000;
  opts.num_particles  = 50;
  opts.step_size      = 0.05;
  opts.temperature    = 0.5;
  opts.use_adagrad    = false;
  opts.seed           = 42;

  GaussianMixtureProblem p(2, 3, 4.0);
  // Find the mode cost (minimum over mode centers)
  double mode_cost = std::numeric_limits<double>::infinity();
  for (const auto& m : p.Modes())
    mode_cost = std::min(mode_cost, p.Evaluate(m.mean));

  auto result = SVGDSolver(opts).Solve(p, Vector::Zero(2));
  // Best particle should be near a mode
  EXPECT_LT(result.best_cost, mode_cost + 0.5);
}
