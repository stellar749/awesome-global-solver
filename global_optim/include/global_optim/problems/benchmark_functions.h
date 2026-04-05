#pragma once

#include "global_optim/core/problem.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

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

// ============================================================
// L2 Global Optimization Functions
// Multimodal, non-convex — designed to challenge global search.
// ============================================================

// Rastrigin: f(x) = 10n + Σ [x_i² - 10 cos(2π x_i)]
// Global min: x* = 0, f* = 0  |  Bounds: [-5.12, 5.12]
// Exponentially many local minima from the cosine perturbation.
class RastriginProblem : public Problem {
 public:
  explicit RastriginProblem(int dim) : dim_(dim) {}

  double Evaluate(const Vector& x) const override {
    double cost = 10.0 * dim_;
    for (int i = 0; i < dim_; ++i)
      cost += x[i] * x[i] - 10.0 * std::cos(2.0 * M_PI * x[i]);
    return cost;
  }

  Vector Gradient(const Vector& x) const override {
    Vector g(dim_);
    for (int i = 0; i < dim_; ++i)
      g[i] = 2.0 * x[i] + 20.0 * M_PI * std::sin(2.0 * M_PI * x[i]);
    return g;
  }

  bool HasGradient() const override { return true; }
  int Dimension() const override { return dim_; }

  Vector LowerBound() const override { return Vector::Constant(dim_, -5.12); }
  Vector UpperBound() const override { return Vector::Constant(dim_,  5.12); }

 private:
  int dim_;
};

// Ackley: f(x) = -20 exp(-0.2 √(||x||²/n)) - exp(Σcos(2πx_i)/n) + 20 + e
// Global min: x* = 0, f* = 0  |  Bounds: [-32.768, 32.768]
// Nearly flat outer region + many local minima near origin.
class AckleyProblem : public Problem {
 public:
  explicit AckleyProblem(int dim) : dim_(dim) {}

  double Evaluate(const Vector& x) const override {
    double sq_sum = x.squaredNorm();
    double cos_sum = 0.0;
    for (int i = 0; i < dim_; ++i)
      cos_sum += std::cos(2.0 * M_PI * x[i]);
    return -20.0 * std::exp(-0.2 * std::sqrt(sq_sum / dim_))
           - std::exp(cos_sum / dim_)
           + 20.0 + std::exp(1.0);
  }

  int Dimension() const override { return dim_; }
  Vector LowerBound() const override { return Vector::Constant(dim_, -32.768); }
  Vector UpperBound() const override { return Vector::Constant(dim_,  32.768); }

 private:
  int dim_;
};

// Schwefel: f(x) = 418.9829 n - Σ x_i sin(√|x_i|)
// Global min: x_i* = 420.9687, f* ≈ 0  |  Bounds: [-500, 500]
// Deceptive: global optimum lies far from next-best solutions.
class SchwefelProblem : public Problem {
 public:
  explicit SchwefelProblem(int dim) : dim_(dim) {}

  double Evaluate(const Vector& x) const override {
    double cost = 418.9829 * dim_;
    for (int i = 0; i < dim_; ++i)
      cost -= x[i] * std::sin(std::sqrt(std::abs(x[i])));
    return cost;
  }

  int Dimension() const override { return dim_; }
  Vector LowerBound() const override { return Vector::Constant(dim_, -500.0); }
  Vector UpperBound() const override { return Vector::Constant(dim_,  500.0); }

 private:
  int dim_;
};

// Double-Rosenbrock: f(z) = min{Rosenbrock(-z-10), 5 + Rosenbrock((z-10)/4)}
// (NES paper, Section 6.4; designed for 2D but works in any dim)
// Global min: z* ≈ -11, f* = 0  |  Local min: z* ≈ 14, f* = 5
// Deceptive: the gradient near z=0 points toward the local minimum.
class DoubleRosenbrockProblem : public Problem {
 public:
  explicit DoubleRosenbrockProblem(int dim = 2) : dim_(dim) {
    if (dim < 2) throw std::invalid_argument("DoubleRosenbrock requires dim >= 2");
  }

  double Evaluate(const Vector& z) const override {
    return std::min(rosenbrock(-z.array() - 10.0),
                    5.0 + rosenbrock((z.array() - 10.0) / 4.0));
  }

  int Dimension() const override { return dim_; }

 private:
  int dim_;

  double rosenbrock(const Eigen::ArrayXd& x) const {
    double cost = 0.0;
    for (int i = 0; i < dim_ - 1; ++i) {
      double a = x[i + 1] - x[i] * x[i];
      double b = 1.0 - x[i];
      cost += 100.0 * a * a + b * b;
    }
    return cost;
  }
};

// GaussianMixture: f(x) = -log(Σ_k w_k N(x; μ_k, σ_k² I))
// Global min at the dominant mode (largest w_k * σ_k^{-d}).
// Ideal for testing multi-mode discovery: SVGD should find all modes.
//
// Default 2D config (3 modes, equal weights):
//   μ = {(0,0), (4,0), (0,4)}, σ = {1, 1, 1}
// For arbitrary dim: modes placed along axis-aligned directions.
class GaussianMixtureProblem : public Problem {
 public:
  struct Mode {
    Vector mean;
    double sigma;   // isotropic std dev
    double weight;  // unnormalized; internally normalized
  };

  // Build a d-dimensional GMM with k_modes equally-weighted modes.
  // Mode centers are placed deterministically for reproducibility.
  explicit GaussianMixtureProblem(int dim, int k_modes = 3, double spread = 4.0)
      : dim_(dim) {
    double w = 1.0 / k_modes;
    for (int k = 0; k < k_modes; ++k) {
      Mode m;
      m.mean = Vector::Zero(dim);
      // Place modes along different axes (cycling through dimensions)
      if (k > 0) m.mean[k % dim] = spread * (k % 2 == 0 ? 1.0 : -1.0);
      m.sigma = 1.0;
      m.weight = w;
      modes_.push_back(m);
    }
  }

  // Custom mode configuration
  explicit GaussianMixtureProblem(std::vector<Mode> modes)
      : dim_(modes.empty() ? 0 : modes[0].mean.size()),
        modes_(std::move(modes)) {
    double wsum = 0;
    for (auto& m : modes_) wsum += m.weight;
    for (auto& m : modes_) m.weight /= wsum;
  }

  double Evaluate(const Vector& x) const override {
    double mix = 0.0;
    for (const auto& m : modes_) {
      double d2 = (x - m.mean).squaredNorm();
      double log_norm = -0.5 * d2 / (m.sigma * m.sigma)
                      - dim_ * std::log(m.sigma)
                      - 0.5 * dim_ * std::log(2.0 * M_PI);
      mix += m.weight * std::exp(log_norm);
    }
    return -std::log(mix + 1e-300);
  }

  Vector Gradient(const Vector& x) const override {
    // grad f = -grad log p = -(1/p) * grad p
    // grad p = Σ_k w_k N_k * (-(x-μ_k)/σ_k²)
    double p = 0.0;
    Vector grad_p = Vector::Zero(dim_);
    for (const auto& m : modes_) {
      double d2 = (x - m.mean).squaredNorm();
      double log_norm = -0.5 * d2 / (m.sigma * m.sigma)
                      - dim_ * std::log(m.sigma)
                      - 0.5 * dim_ * std::log(2.0 * M_PI);
      double nk = m.weight * std::exp(log_norm);
      p += nk;
      grad_p += nk * (-(x - m.mean) / (m.sigma * m.sigma));
    }
    return -grad_p / (p + 1e-300);
  }

  bool HasGradient() const override { return true; }
  int Dimension() const override { return dim_; }

  const std::vector<Mode>& Modes() const { return modes_; }

 private:
  int dim_;
  std::vector<Mode> modes_;
};

// Griewank: f(x) = 1 + Σ x_i²/4000 - Π cos(x_i / √i)
// Global min: x* = 0, f* = 0  |  Bounds: [-600, 600]
// Regular grid of local minima; coordinate product makes them hard to locate.
class GriewankProblem : public Problem {
 public:
  explicit GriewankProblem(int dim) : dim_(dim) {}

  double Evaluate(const Vector& x) const override {
    double sum = x.squaredNorm() / 4000.0;
    double prod = 1.0;
    for (int i = 0; i < dim_; ++i)
      prod *= std::cos(x[i] / std::sqrt(static_cast<double>(i + 1)));
    return 1.0 + sum - prod;
  }

  Vector Gradient(const Vector& x) const override {
    // Precompute per-dim cosines and the full product
    std::vector<double> ci(dim_), si(dim_), ri(dim_);
    double prod_all = 1.0;
    for (int i = 0; i < dim_; ++i) {
      ri[i] = std::sqrt(static_cast<double>(i + 1));
      ci[i] = std::cos(x[i] / ri[i]);
      si[i] = std::sin(x[i] / ri[i]);
      prod_all *= ci[i];
    }
    Vector g(dim_);
    for (int i = 0; i < dim_; ++i) {
      double prod_except_i = (ci[i] != 0.0) ? prod_all / ci[i] : 0.0;
      g[i] = x[i] / 2000.0 + prod_except_i * si[i] / ri[i];
    }
    return g;
  }

  bool HasGradient() const override { return true; }
  int Dimension() const override { return dim_; }

  Vector LowerBound() const override { return Vector::Constant(dim_, -600.0); }
  Vector UpperBound() const override { return Vector::Constant(dim_,  600.0); }

 private:
  int dim_;
};

// RandomBasin: purely stochastic multimodal landscape from NES paper §6.4.
//
// f(z) = 1 - 0.9·R(⌊z/10⌋) - 0.1·R(⌊z⌋) · ∏_i sin²(π z_i)^{1/(200d)}
//
// R(k) maps an integer d-vector k to [0,1] via a deterministic hash.
// Each unit hypercube [k, k+1)^d forms one basin; the landscape has no
// exploitable global structure — only random attractiveness per basin.
// Global minimum ≈ 0 (when both R values ≈ 1 and sin²-product ≈ 1).
// HasGradient() = false; SVGD cannot run on this problem.
//
// Domain: [-50, 50]^d  (100 fine basins per dimension).
class RandomBasinProblem : public Problem {
 public:
  explicit RandomBasinProblem(int dim) : dim_(dim) {}

  double Evaluate(const Vector& z) const override {
    // Coarse-scale basin (scale 10)
    std::vector<int> k_coarse(dim_);
    for (int i = 0; i < dim_; ++i)
      k_coarse[i] = static_cast<int>(std::floor(z[i] / 10.0));

    // Fine-scale basin (scale 1)
    std::vector<int> k_fine(dim_);
    for (int i = 0; i < dim_; ++i)
      k_fine[i] = static_cast<int>(std::floor(z[i]));

    // sin² smoothing term: product over dims of sin²(π z_i)^{1/(200d)}
    // Use log-sum to avoid underflow
    double log_prod = 0.0;
    const double exponent = 1.0 / (200.0 * dim_);
    for (int i = 0; i < dim_; ++i) {
      double s2 = std::sin(M_PI * z[i]);
      s2 = s2 * s2;
      log_prod += exponent * std::log(s2 + 1e-300);
    }
    double sin_prod = std::exp(log_prod);

    return 1.0 - 0.9 * hash_to_01(k_coarse) - 0.1 * hash_to_01(k_fine) * sin_prod;
  }

  bool HasGradient() const override { return false; }
  int Dimension() const override { return dim_; }

  Vector LowerBound() const override { return Vector::Constant(dim_, -50.0); }
  Vector UpperBound() const override { return Vector::Constant(dim_,  50.0); }

 private:
  int dim_;

  // Deterministic hash: integer d-vector → [0, 1].
  // Uses FNV-1a mix on each component.
  static double hash_to_01(const std::vector<int>& k) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < static_cast<int>(k.size()); ++i) {
      // Mix in both the index and the value to make different dimensions interact
      h ^= static_cast<uint64_t>(k[i]) ^ (static_cast<uint64_t>(i) << 32);
      h *= 0x100000001b3ULL;
    }
    // Finalizer (based on MurmurHash3 fmix64)
    h ^= (h >> 33);
    h *= 0xff51afd7ed558ccdULL;
    h ^= (h >> 33);
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= (h >> 33);
    return static_cast<double>(h >> 16) / static_cast<double>(UINT64_C(1) << 48);
  }
};

}  // namespace global_optim
