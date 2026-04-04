#include "global_optim/problems/benchmark_functions.h"

#include <gtest/gtest.h>
#include <cmath>

using namespace global_optim;

// ─── Sphere ──────────────────────────────────────────────────────────────────

TEST(Sphere, GlobalMinimumIsZero) {
  SphereProblem p(5);
  EXPECT_DOUBLE_EQ(p.Evaluate(Vector::Zero(5)), 0.0);
}

TEST(Sphere, KnownValue) {
  SphereProblem p(3);
  Vector x(3); x << 1.0, 2.0, 3.0;
  EXPECT_DOUBLE_EQ(p.Evaluate(x), 14.0);  // 1+4+9
}

TEST(Sphere, Dimension) {
  EXPECT_EQ(SphereProblem(7).Dimension(), 7);
}

TEST(Sphere, GradientAtOrigin) {
  SphereProblem p(4);
  EXPECT_TRUE(p.Gradient(Vector::Zero(4)).isZero());
}

TEST(Sphere, GradientKnownValue) {
  SphereProblem p(3);
  Vector x(3); x << 1.0, 2.0, 3.0;
  Vector g_expected(3); g_expected << 2.0, 4.0, 6.0;
  EXPECT_TRUE(p.Gradient(x).isApprox(g_expected));
}

// ─── Ellipsoid ────────────────────────────────────────────────────────────────

TEST(Ellipsoid, GlobalMinimumIsZero) {
  EllipsoidProblem p(5);
  EXPECT_DOUBLE_EQ(p.Evaluate(Vector::Zero(5)), 0.0);
}

TEST(Ellipsoid, KnownValue) {
  // f([1,1,1]) = 1*1 + 2*1 + 3*1 = 6
  EllipsoidProblem p(3);
  Vector x = Vector::Ones(3);
  EXPECT_DOUBLE_EQ(p.Evaluate(x), 6.0);
}

TEST(Ellipsoid, Dimension) {
  EXPECT_EQ(EllipsoidProblem(10).Dimension(), 10);
}

TEST(Ellipsoid, GradientKnownValue) {
  EllipsoidProblem p(3);
  Vector x = Vector::Ones(3);
  // g_i = 2*(i+1)*x_i  →  [2, 4, 6]
  Vector g_expected(3); g_expected << 2.0, 4.0, 6.0;
  EXPECT_TRUE(p.Gradient(x).isApprox(g_expected));
}

// ─── Rosenbrock ───────────────────────────────────────────────────────────────

TEST(Rosenbrock, GlobalMinimumIsZero) {
  RosenbrockProblem p(5);
  EXPECT_DOUBLE_EQ(p.Evaluate(Vector::Ones(5)), 0.0);
}

TEST(Rosenbrock, KnownValue2D) {
  // f([0,0]) = 100*(0-0)^2 + (1-0)^2 = 1
  RosenbrockProblem p(2);
  Vector x = Vector::Zero(2);
  EXPECT_DOUBLE_EQ(p.Evaluate(x), 1.0);
}

TEST(Rosenbrock, Dimension) {
  EXPECT_EQ(RosenbrockProblem(6).Dimension(), 6);
}

TEST(Rosenbrock, GradientAtMinimum) {
  // At x* = (1,...,1), gradient should be zero
  RosenbrockProblem p(4);
  Vector g = p.Gradient(Vector::Ones(4));
  EXPECT_TRUE(g.isZero(1e-12));
}

TEST(Rosenbrock, RequiresDimAtLeast2) {
  EXPECT_THROW(RosenbrockProblem(1), std::invalid_argument);
}

// ─── Cigar ────────────────────────────────────────────────────────────────────

TEST(Cigar, GlobalMinimumIsZero) {
  CigarProblem p(5);
  EXPECT_DOUBLE_EQ(p.Evaluate(Vector::Zero(5)), 0.0);
}

TEST(Cigar, KnownValue) {
  // f([1, 1, 1]) = 1 + 1e6 + 1e6 = 2000001
  CigarProblem p(3);
  Vector x = Vector::Ones(3);
  EXPECT_DOUBLE_EQ(p.Evaluate(x), 2000001.0);
}

TEST(Cigar, Dimension) {
  EXPECT_EQ(CigarProblem(8).Dimension(), 8);
}

TEST(Cigar, GradientKnownValue) {
  CigarProblem p(3);
  Vector x = Vector::Ones(3);
  Vector g_expected(3); g_expected << 2.0, 2e6, 2e6;
  EXPECT_TRUE(p.Gradient(x).isApprox(g_expected));
}

TEST(Cigar, IllConditioned) {
  // x=[1,0,...,0] should have much lower cost than x=[0,1,0,...,0]
  CigarProblem p(4);
  Vector xe1 = Vector::Zero(4); xe1[0] = 1.0;
  Vector xe2 = Vector::Zero(4); xe2[1] = 1.0;
  EXPECT_LT(p.Evaluate(xe1), p.Evaluate(xe2));
}
