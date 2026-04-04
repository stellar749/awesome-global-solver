#include "global_optim/solvers/cmaes.h"
#include "global_optim/problems/benchmark_functions.h"

#include <gtest/gtest.h>

using namespace global_optim;

// Helper: run solver and return best cost
static double Solve(CMAESOptions opts, const Problem& p, const Vector& x0) {
  return CMAESSolver(opts).Solve(p, x0).best_cost;
}

// ── Sphere ────────────────────────────────────────────────────────────────────

TEST(CMAES, SphereConverges_5D) {
  CMAESOptions opts;
  opts.max_iterations = 500;
  SphereProblem p(5);
  Vector x0 = Vector::Ones(5) * 3.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-10);
}

TEST(CMAES, SphereConverges_20D) {
  CMAESOptions opts;
  opts.max_iterations = 2000;
  SphereProblem p(20);
  Vector x0 = Vector::Ones(20) * 3.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-8);
}

TEST(CMAES, SphereResultHasBestX) {
  CMAESOptions opts;
  opts.max_iterations = 300;
  SphereProblem p(5);
  Vector x0 = Vector::Ones(5) * 2.0;
  auto result = CMAESSolver(opts).Solve(p, x0);
  // best_x should evaluate close to best_cost
  EXPECT_NEAR(p.Evaluate(result.best_x), result.best_cost, 1e-12);
}

// ── Ellipsoid ─────────────────────────────────────────────────────────────────

TEST(CMAES, EllipsoidConverges) {
  CMAESOptions opts;
  opts.max_iterations = 1000;
  EllipsoidProblem p(10);
  Vector x0 = Vector::Ones(10) * 2.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-8);
}

// ── Rosenbrock ─────────────────────────────────────────────────────────────────

TEST(CMAES, RosenbrockConverges_5D) {
  CMAESOptions opts;
  opts.max_iterations = 2000;
  opts.sigma0 = 0.5;
  RosenbrockProblem p(5);
  Vector x0 = Vector::Zero(5);  // start away from x*=(1,...,1)
  EXPECT_LT(Solve(opts, p, x0), 1e-8);
}

// ── Cigar ─────────────────────────────────────────────────────────────────────

TEST(CMAES, CigarConverges) {
  CMAESOptions opts;
  opts.max_iterations = 1000;
  CigarProblem p(10);
  Vector x0 = Vector::Ones(10);
  EXPECT_LT(Solve(opts, p, x0), 1e-6);
}

// ── Metadata ─────────────────────────────────────────────────────────────────

TEST(CMAES, ResultMetadata) {
  CMAESOptions opts;
  opts.max_iterations = 100;
  SphereProblem p(5);
  auto result = CMAESSolver(opts).Solve(p, Vector::Ones(5));
  EXPECT_GT(result.num_evaluations, 0);
  EXPECT_GT(result.num_iterations, 0);
  EXPECT_GT(result.elapsed_time_ms, 0.0);
  EXPECT_FALSE(result.cost_history.empty());
  // Cost history should be non-increasing
  for (size_t i = 1; i < result.cost_history.size(); ++i)
    EXPECT_LE(result.cost_history[i], result.cost_history[i - 1] + 1e-15);
}

TEST(CMAES, CostTargetEarlyStop) {
  CMAESOptions opts;
  opts.max_iterations = 10000;
  opts.cost_target = 1e-4;
  SphereProblem p(5);
  auto result = CMAESSolver(opts).Solve(p, Vector::Ones(5) * 3.0);
  EXPECT_LE(result.best_cost, 1e-4);
  EXPECT_LT(result.num_iterations, 10000);  // stopped early
}

TEST(CMAES, Name) {
  EXPECT_EQ(CMAESSolver().Name(), "CMA-ES");
}
