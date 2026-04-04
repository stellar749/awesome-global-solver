#pragma once

#include "global_optim/core/problem.h"
#include "global_optim/core/result.h"
#include "global_optim/core/solver.h"
#include "global_optim/core/random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace global_optim {

struct SVGDOptions : SolverOptions {
  int num_particles = 50;    // number of particles
  double step_size = 0.01;   // base learning rate ε
  double temperature = 1.0;  // τ: target p(x) ∝ exp(-f(x)/τ)
  bool use_adagrad = true;   // adaptive learning rate (recommended)
  double bandwidth = -1.0;   // h < 0 → median heuristic (adaptive)
  double init_std = 1.0;     // particle initialization spread around x0
};

// Stein Variational Gradient Descent (Liu & Wang, NeurIPS 2016).
//
// Drives n particles {x_i} to approximate the Boltzmann distribution
//   p(x) ∝ exp(-f(x) / τ)
// via the Stein operator with RBF kernel. Requires Problem::Gradient().
//
// SVGD update:
//   φ(x_i) = (1/n) Σ_j [ k(x_j, x_i) ∇log p(x_j) + ∇_{x_j} k(x_j, x_i) ]
//   x_i   ← x_i + ε · φ(x_i)
//
// where ∇log p(x) = -∇f(x)/τ  (score function)
// and k(x, y) = exp(-||x-y||² / h)  (RBF kernel)
// with bandwidth h = median²(||x_i-x_j||²) / log(n)  (median heuristic)
//
// Returns the particle with lowest cost as best_x.
class SVGDSolver : public Solver {
 public:
  explicit SVGDSolver(SVGDOptions opts = {}) : opts_(opts) {}

  std::string Name() const override { return "SVGD"; }

  SolverResult Solve(const Problem& problem, const Vector& x0) override {
    if (!problem.HasGradient())
      throw std::runtime_error("SVGD requires Problem::Gradient()");

    auto t0 = std::chrono::steady_clock::now();
    const int d = problem.Dimension();
    const int n = opts_.num_particles;
    RandomEngine rng(opts_.seed);

    // ── Initialize particles around x0 ────────────────────────────────────
    // Use small isotropic spread so particles start exploring immediately
    Matrix X(n, d);  // particles: rows
    for (int i = 0; i < n; ++i)
      X.row(i) = x0 + opts_.init_std * rng.RandNVector(d).transpose();

    // AdaGrad accumulated gradient squared (per particle per dim)
    Matrix G_hist = Matrix::Ones(n, d) * 1e-8;  // small init avoids div-by-zero

    // ── Bookkeeping ─────────────────────────────────────────────────────────
    SolverResult result;
    result.num_evaluations = 0;

    auto eval_best = [&]() -> std::pair<double, int> {
      double best = std::numeric_limits<double>::infinity();
      int bi = 0;
      for (int i = 0; i < n; ++i) {
        double c = problem.Evaluate(X.row(i).transpose());
        ++result.num_evaluations;
        if (c < best) { best = c; bi = i; }
      }
      return {best, bi};
    };

    auto [init_cost, init_bi] = eval_best();
    result.best_cost = init_cost;
    result.best_x = X.row(init_bi).transpose();

    for (int iter = 0; iter < opts_.max_iterations; ++iter) {
      if (result.num_evaluations >= opts_.max_evaluations) break;
      if (result.best_cost <= opts_.cost_target) break;

      // ── 1. Compute score functions: ∇log p(x_i) = -∇f(x_i)/τ ────────────
      Matrix scores(n, d);
      for (int i = 0; i < n; ++i) {
        Vector xi = X.row(i).transpose();
        scores.row(i) = (-1.0 / opts_.temperature) * problem.Gradient(xi).transpose();
      }
      // Note: gradient evaluations counted separately (not re-evaluating f here)

      // ── 2. Bandwidth via median heuristic ─────────────────────────────────
      double h;
      if (opts_.bandwidth > 0) {
        h = opts_.bandwidth;
      } else {
        // Collect pairwise squared distances
        std::vector<double> dists;
        dists.reserve(n * (n - 1) / 2);
        for (int i = 0; i < n; ++i)
          for (int j = i + 1; j < n; ++j)
            dists.push_back((X.row(i) - X.row(j)).squaredNorm());

        if (!dists.empty()) {
          auto mid = dists.begin() + dists.size() / 2;
          std::nth_element(dists.begin(), mid, dists.end());
          double med2 = *mid;
          h = med2 / std::max(1.0, std::log(static_cast<double>(n)));
          h = std::max(h, 1e-10);
        } else {
          h = 1.0;
        }
      }

      // ── 3. SVGD update direction ─────────────────────────────────────────
      // φ(x_i) = (1/n) Σ_j [ k_ij * score_j + ∇_{x_j} k_ij ]
      // ∇_{x_j} k(x_j, x_i) = (2/h) * (x_i - x_j) * k_ij
      Matrix phi(n, d);
      phi.setZero();

      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          Vector diff = X.row(i).transpose() - X.row(j).transpose();
          double k_ij = std::exp(-diff.squaredNorm() / h);
          // kernel gradient w.r.t. x_j: +(2/h)*(x_i - x_j)*k_ij
          phi.row(i) += k_ij * scores.row(j)
                      + (2.0 / h) * k_ij * diff.transpose();
        }
        phi.row(i) /= n;
      }

      // ── 4. Update particles ───────────────────────────────────────────────
      if (opts_.use_adagrad) {
        G_hist += phi.cwiseProduct(phi);
        X += opts_.step_size * phi.cwiseQuotient(
            (G_hist.array().sqrt() + 1e-8).matrix());
      } else {
        X += opts_.step_size * phi;
      }

      // ── 5. Track best particle ────────────────────────────────────────────
      auto [iter_best, iter_bi] = eval_best();
      if (iter_best < result.best_cost) {
        result.best_cost = iter_best;
        result.best_x = X.row(iter_bi).transpose();
      }
      result.cost_history.push_back(result.best_cost);

      if (opts_.verbose)
        printf("[SVGD] iter %4d  evals %6d  best %.6e  h %.4e\n",
               iter, result.num_evaluations, result.best_cost, h);
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.num_iterations = static_cast<int>(result.cost_history.size());
    return result;
  }

 private:
  SVGDOptions opts_;
};

}  // namespace global_optim
