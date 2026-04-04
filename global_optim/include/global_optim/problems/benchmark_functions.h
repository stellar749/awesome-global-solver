#pragma once

#include "global_optim/core/problem.h"

#include <cmath>
#include <stdexcept>

namespace global_optim {

// ============================================================
// L1 Correctness Functions
// All have a unique global minimum at x* = 0 with f(x*) = 0
// (except Rosenbrock: x* = 1, f(x*) = 0)
// ============================================================

// Sphere: f(x) = sum(x_i^2)
// Global min: x* = 0, f* = 0
// Separable, unimodal, isotropic baseline.
class SphereProblem : public Problem {
 public:
  explicit SphereProblem(int dim) : dim_(dim) {}

  double Evaluate(const Vector& x) const override {
    return x.squaredNorm();
  }

  Vector Gradient(const Vector& x) const override {
    return 2.0 * x;
  }

  bool HasGradient() const override { return true; }

  int Dimension() const override { return dim_; }

 private:
  int dim_;
};

// Ellipsoid: f(x) = sum(i * x_i^2)  (i = 1..n, 1-indexed)
// Global min: x* = 0, f* = 0
// Ill-conditioned separable function; tests step-size adaptation.
class EllipsoidProblem : public Problem {
 public:
  explicit EllipsoidProblem(int dim) : dim_(dim) {}

  double Evaluate(const Vector& x) const override {
    double cost = 0.0;
    for (int i = 0; i < dim_; ++i)
      cost += static_cast<double>(i + 1) * x[i] * x[i];
    return cost;
  }

  Vector Gradient(const Vector& x) const override {
    Vector g(dim_);
    for (int i = 0; i < dim_; ++i)
      g[i] = 2.0 * static_cast<double>(i + 1) * x[i];
    return g;
  }

  bool HasGradient() const override { return true; }

  int Dimension() const override { return dim_; }

 private:
  int dim_;
};

// Rosenbrock: f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
// Global min: x* = (1,...,1), f* = 0
// Narrow curved valley; classic test for covariance learning.
class RosenbrockProblem : public Problem {
 public:
  explicit RosenbrockProblem(int dim) : dim_(dim) {
    if (dim < 2)
      throw std::invalid_argument("Rosenbrock requires dim >= 2");
  }

  double Evaluate(const Vector& x) const override {
    double cost = 0.0;
    for (int i = 0; i < dim_ - 1; ++i) {
      double a = x[i + 1] - x[i] * x[i];
      double b = 1.0 - x[i];
      cost += 100.0 * a * a + b * b;
    }
    return cost;
  }

  Vector Gradient(const Vector& x) const override {
    Vector g = Vector::Zero(dim_);
    for (int i = 0; i < dim_ - 1; ++i) {
      double a = x[i + 1] - x[i] * x[i];
      g[i]     += -400.0 * a * x[i] - 2.0 * (1.0 - x[i]);
      g[i + 1] += 200.0 * a;
    }
    return g;
  }

  bool HasGradient() const override { return true; }

  int Dimension() const override { return dim_; }

 private:
  int dim_;
};

// Cigar (Discus variant): f(x) = x_0^2 + 1e6 * sum_{i=1}^{n-1} x_i^2
// Global min: x* = 0, f* = 0
// Extreme axis-aligned ill-conditioning (condition number = 1e6).
// Tests whether the solver can discover the single easy dimension.
class CigarProblem : public Problem {
 public:
  explicit CigarProblem(int dim) : dim_(dim) {}

  double Evaluate(const Vector& x) const override {
    double cost = x[0] * x[0];
    for (int i = 1; i < dim_; ++i)
      cost += 1e6 * x[i] * x[i];
    return cost;
  }

  Vector Gradient(const Vector& x) const override {
    Vector g(dim_);
    g[0] = 2.0 * x[0];
    for (int i = 1; i < dim_; ++i)
      g[i] = 2e6 * x[i];
    return g;
  }

  bool HasGradient() const override { return true; }

  int Dimension() const override { return dim_; }

 private:
  int dim_;
};

}  // namespace global_optim
