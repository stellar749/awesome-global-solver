#include "global_optim/solvers/xnes.h"
#include "global_optim/problems/benchmark_functions.h"

#include <gtest/gtest.h>

using namespace global_optim;

static double Solve(XNESOptions opts, const Problem& p, const Vector& x0) {
  return XNESSolver(opts).Solve(p, x0).best_cost;
}

// ── Sphere ────────────────────────────────────────────────────────────────────

TEST(XNES, SphereConverges_5D) {
  XNESOptions opts;
  opts.max_iterations = 1000;
  SphereProblem p(5);
  Vector x0 = Vector::Ones(5) * 3.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-8);
}

TEST(XNES, SphereConverges_10D) {
  XNESOptions opts;
  opts.max_iterations = 2000;
  SphereProblem p(10);
  Vector x0 = Vector::Ones(10) * 3.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-6);
}

TEST(XNES, SphereResultHasBestX) {
  XNESOptions opts;
  opts.max_iterations = 500;
  SphereProblem p(5);
  auto result = XNESSolver(opts).Solve(p, Vector::Ones(5) * 2.0);
  EXPECT_NEAR(p.Evaluate(result.best_x), result.best_cost, 1e-12);
}

// ── Ellipsoid ─────────────────────────────────────────────────────────────────

TEST(XNES, EllipsoidConverges) {
  XNESOptions opts;
  opts.max_iterations = 2000;
  EllipsoidProblem p(10);
  Vector x0 = Vector::Ones(10) * 2.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-6);
}

// ── Rosenbrock ─────────────────────────────────────────────────────────────────

TEST(XNES, RosenbrockConverges_5D) {
  XNESOptions opts;
  opts.max_iterations = 3000;
  opts.sigma0 = 1.0;
  RosenbrockProblem p(5);
  Vector x0 = Vector::Zero(5);
  EXPECT_LT(Solve(opts, p, x0), 1e-6);
}

// ── Cigar ─────────────────────────────────────────────────────────────────────

TEST(XNES, CigarConverges) {
  XNESOptions opts;
  opts.max_iterations = 2000;
  CigarProblem p(5);
  Vector x0 = Vector::Ones(5);
  EXPECT_LT(Solve(opts, p, x0), 1e-4);
}

// ── Metadata ─────────────────────────────────────────────────────────────────

TEST(XNES, ResultMetadata) {
  XNESOptions opts;
  opts.max_iterations = 100;
  SphereProblem p(5);
  auto result = XNESSolver(opts).Solve(p, Vector::Ones(5));
  EXPECT_GT(result.num_evaluations, 0);
  EXPECT_GT(result.num_iterations, 0);
  EXPECT_GT(result.elapsed_time_ms, 0.0);
  EXPECT_FALSE(result.cost_history.empty());
}

TEST(XNES, CostTargetEarlyStop) {
  XNESOptions opts;
  opts.max_iterations = 10000;
  opts.cost_target = 1e-4;
  SphereProblem p(5);
  auto result = XNESSolver(opts).Solve(p, Vector::Ones(5) * 3.0);
  EXPECT_LE(result.best_cost, 1e-4);
  EXPECT_LT(result.num_iterations, 10000);
}

TEST(XNES, Name) {
  EXPECT_EQ(XNESSolver().Name(), "xNES");
}
