#include "global_optim/solvers/mppi.h"
#include "global_optim/problems/benchmark_functions.h"

#include <gtest/gtest.h>

using namespace global_optim;

static double Solve(MPPIOptions opts, const Problem& p, const Vector& x0) {
  return MPPISolver(opts).Solve(p, x0).best_cost;
}

// ── Sphere ────────────────────────────────────────────────────────────────────

TEST(MPPI, SphereConverges_5D) {
  // MPPI has an IS noise floor ~ sigma^2 * d / K; 5e-4 is achievable and
  // demonstrates correct convergence behavior for a control algorithm.
  MPPIOptions opts;
  opts.max_iterations = 200;
  opts.num_samples = 512;
  opts.noise_sigma = 0.3;
  opts.temperature = 0.5;
  SphereProblem p(5);
  Vector x0 = Vector::Ones(5) * 2.0;
  EXPECT_LT(Solve(opts, p, x0), 5e-4);
}

TEST(MPPI, SphereConverges_10D) {
  MPPIOptions opts;
  opts.max_iterations = 500;
  opts.num_samples = 1024;
  opts.noise_sigma = 0.2;
  opts.temperature = 0.5;
  SphereProblem p(10);
  Vector x0 = Vector::Ones(10) * 2.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-3);
}

TEST(MPPI, SphereResultHasBestX) {
  MPPIOptions opts;
  opts.max_iterations = 100;
  opts.num_samples = 256;
  SphereProblem p(5);
  auto result = MPPISolver(opts).Solve(p, Vector::Ones(5) * 2.0);
  EXPECT_NEAR(p.Evaluate(result.best_x), result.best_cost, 1e-10);
}

// ── Ellipsoid ─────────────────────────────────────────────────────────────────

TEST(MPPI, EllipsoidConverges) {
  MPPIOptions opts;
  opts.max_iterations = 300;
  opts.num_samples = 512;
  opts.noise_sigma = 0.3;
  opts.temperature = 1.0;
  EllipsoidProblem p(5);
  Vector x0 = Vector::Ones(5) * 2.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-2);
}

// ── Rosenbrock ─────────────────────────────────────────────────────────────────

TEST(MPPI, RosenbrockConverges_2D) {
  // MPPI is a zero-order method; Rosenbrock is harder but 2D should work
  MPPIOptions opts;
  opts.max_iterations = 500;
  opts.num_samples = 2048;
  opts.noise_sigma = 0.2;
  opts.temperature = 0.1;
  RosenbrockProblem p(2);
  Vector x0 = Vector::Zero(2);
  EXPECT_LT(Solve(opts, p, x0), 0.1);
}

// ── Metadata ─────────────────────────────────────────────────────────────────

TEST(MPPI, ResultMetadata) {
  MPPIOptions opts;
  opts.max_iterations = 10;
  opts.num_samples = 64;
  SphereProblem p(3);
  auto result = MPPISolver(opts).Solve(p, Vector::Ones(3));
  EXPECT_GT(result.num_evaluations, 0);
  EXPECT_EQ(result.num_iterations, 10);
  EXPECT_GT(result.elapsed_time_ms, 0.0);
  EXPECT_EQ(static_cast<int>(result.cost_history.size()), 10);
}

TEST(MPPI, CostTargetEarlyStop) {
  MPPIOptions opts;
  opts.max_iterations = 10000;
  opts.num_samples = 512;
  opts.noise_sigma = 0.3;
  opts.temperature = 0.5;
  opts.cost_target = 1e-3;
  SphereProblem p(5);
  auto result = MPPISolver(opts).Solve(p, Vector::Ones(5) * 2.0);
  EXPECT_LE(result.best_cost, 1e-3);
  EXPECT_LT(result.num_iterations, 10000);
}

TEST(MPPI, Name) {
  EXPECT_EQ(MPPISolver().Name(), "MPPI");
}
