#include "global_optim/solvers/svgd.h"
#include "global_optim/problems/benchmark_functions.h"

#include <gtest/gtest.h>

using namespace global_optim;

static double Solve(SVGDOptions opts, const Problem& p, const Vector& x0) {
  return SVGDSolver(opts).Solve(p, x0).best_cost;
}

// ── Sphere ────────────────────────────────────────────────────────────────────
// Note: SVGD maintains a finite-width Boltzmann distribution (not a point mass),
// so convergence precision is bounded by temperature. We use small tau=0.01
// and no AdaGrad to verify correct convergence behavior.

TEST(SVGD, SphereConverges_5D) {
  SVGDOptions opts;
  opts.max_iterations = 2000;
  opts.step_size = 0.01;
  opts.temperature = 0.01;
  opts.num_particles = 50;
  opts.use_adagrad = false;
  SphereProblem p(5);
  Vector x0 = Vector::Ones(5) * 2.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-3);
}

TEST(SVGD, SphereConverges_10D) {
  SVGDOptions opts;
  opts.max_iterations = 2000;
  opts.step_size = 0.01;
  opts.temperature = 0.01;
  opts.num_particles = 50;
  opts.use_adagrad = false;
  SphereProblem p(10);
  Vector x0 = Vector::Ones(10) * 2.0;
  EXPECT_LT(Solve(opts, p, x0), 1e-2);
}

TEST(SVGD, SphereResultHasBestX) {
  SVGDOptions opts;
  opts.max_iterations = 200;
  opts.num_particles = 10;
  SphereProblem p(5);
  auto result = SVGDSolver(opts).Solve(p, Vector::Ones(5) * 2.0);
  EXPECT_NEAR(p.Evaluate(result.best_x), result.best_cost, 1e-10);
}

// ── Ellipsoid ─────────────────────────────────────────────────────────────────

TEST(SVGD, EllipsoidConverges) {
  // SVGD equilibrium cost is O(tau*n*particles), use relative improvement check.
  SVGDOptions opts;
  opts.max_iterations = 2000;
  opts.step_size = 0.01;
  opts.temperature = 0.01;
  opts.num_particles = 50;
  opts.use_adagrad = false;
  EllipsoidProblem p(5);
  Vector x0 = Vector::Ones(5) * 2.0;
  double initial = p.Evaluate(x0);  // = 1+2+3+4+5 * 4 = 60
  EXPECT_LT(Solve(opts, p, x0), initial * 0.01);  // > 99% cost reduction
}

// ── Rosenbrock ─────────────────────────────────────────────────────────────────

TEST(SVGD, RosenbrockConverges_2D) {
  // Rosenbrock's narrow valley is hard for SVGD; verify substantial reduction.
  SVGDOptions opts;
  opts.max_iterations = 3000;
  opts.step_size = 0.005;
  opts.temperature = 0.01;
  opts.num_particles = 50;
  opts.use_adagrad = false;
  RosenbrockProblem p(2);
  Vector x0(2); x0 << 0.0, 0.0;
  double initial = p.Evaluate(x0);  // = 1.0
  EXPECT_LT(Solve(opts, p, x0), initial * 0.5);  // > 50% cost reduction
}

// ── Gradient required ─────────────────────────────────────────────────────────

TEST(SVGD, ThrowsWithoutGradient) {
  // A Problem that declares HasGradient() = false
  class NoGradProblem : public Problem {
   public:
    double Evaluate(const Vector& x) const override { return x.squaredNorm(); }
    int Dimension() const override { return 3; }
    bool HasGradient() const override { return false; }
  };
  SVGDSolver solver;
  EXPECT_THROW(solver.Solve(NoGradProblem{}, Vector::Zero(3)), std::runtime_error);
}

// ── Metadata ─────────────────────────────────────────────────────────────────

TEST(SVGD, ResultMetadata) {
  SVGDOptions opts;
  opts.max_iterations = 20;
  opts.num_particles = 5;
  SphereProblem p(3);
  auto result = SVGDSolver(opts).Solve(p, Vector::Ones(3));
  EXPECT_GT(result.num_evaluations, 0);
  EXPECT_EQ(result.num_iterations, 20);
  EXPECT_GT(result.elapsed_time_ms, 0.0);
}

TEST(SVGD, CostTargetEarlyStop) {
  SVGDOptions opts;
  opts.max_iterations = 10000;
  opts.step_size = 0.01;
  opts.temperature = 0.01;
  opts.num_particles = 50;
  opts.use_adagrad = false;
  opts.cost_target = 1e-3;
  SphereProblem p(5);
  auto result = SVGDSolver(opts).Solve(p, Vector::Ones(5) * 2.0);
  EXPECT_LE(result.best_cost, 1e-3);
  EXPECT_LT(result.num_iterations, 10000);
}

TEST(SVGD, Name) {
  EXPECT_EQ(SVGDSolver().Name(), "SVGD");
}
