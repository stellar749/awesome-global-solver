#pragma once

#include "global_optim/core/problem.h"
#include "global_optim/core/result.h"
#include "global_optim/core/solver.h"
#include "global_optim/core/random.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

namespace global_optim {

struct CMAESOptions : SolverOptions {
  int lambda = -1;         // population size; default: 4 + floor(3*ln(n))
  int mu = -1;             // parents; default: lambda/2
  double sigma0 = 0.5;     // initial step size
  // adaptive params (< 0 → auto-compute from dimension)
  double c_c = -1;
  double c_sigma = -1;
  double c_1 = -1;
  double c_mu_coeff = -1;
};

// (μ/μ_w, λ)-CMA-ES following Hansen's tutorial (2016).
// Minimizes problem.Evaluate() starting from x0.
class CMAESSolver : public Solver {
 public:
  explicit CMAESSolver(CMAESOptions opts = {}) : opts_(opts) {}

  std::string Name() const override { return "CMA-ES"; }

  SolverResult Solve(const Problem& problem, const Vector& x0) override {
    auto t0 = std::chrono::steady_clock::now();
    const int n = problem.Dimension();
    RandomEngine rng(opts_.seed);

    // ── Hyper-parameters ────────────────────────────────────────────────────
    const int lam = opts_.lambda > 0 ? opts_.lambda
                                     : 4 + static_cast<int>(3.0 * std::log(n));
    const int mu = opts_.mu > 0 ? opts_.mu : lam / 2;

    // Weights
    std::vector<double> w(mu);
    for (int i = 0; i < mu; ++i)
      w[i] = std::log(mu + 0.5) - std::log(i + 1.0);
    double wsum = 0;
    for (double wi : w) wsum += wi;
    for (double& wi : w) wi /= wsum;

    double mueff = 0;
    {
      double w2 = 0;
      for (double wi : w) { mueff += wi; w2 += wi * wi; }
      mueff = mueff * mueff / w2;
    }

    const double c_sigma = opts_.c_sigma > 0 ? opts_.c_sigma
                         : (mueff + 2.0) / (n + mueff + 5.0);
    const double d_sigma = 1.0 + 2.0 * std::max(0.0, std::sqrt((mueff - 1.0) / (n + 1.0)) - 1.0)
                         + c_sigma;
    const double c_c = opts_.c_c > 0 ? opts_.c_c
                     : (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n);
    const double c_1 = opts_.c_1 > 0 ? opts_.c_1
                     : 2.0 / ((n + 1.3) * (n + 1.3) + mueff);
    const double c_mu_raw = 2.0 * (mueff - 2.0 + 1.0 / mueff)
                           / ((n + 2.0) * (n + 2.0) + mueff);
    const double c_mu = opts_.c_mu_coeff > 0
                       ? opts_.c_mu_coeff * c_mu_raw
                       : std::min(1.0 - c_1, c_mu_raw);
    const double chi_n = std::sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));

    // Eigendecomposition refresh interval
    const int eigen_interval = std::max(1, static_cast<int>(
        1.0 / (10.0 * n * (c_1 + c_mu))));

    // ── State ────────────────────────────────────────────────────────────────
    Vector m = x0;
    double sigma = opts_.sigma0;
    Matrix C = Matrix::Identity(n, n);
    Vector p_c = Vector::Zero(n);
    Vector p_sigma = Vector::Zero(n);

    // Eigen decomposition of C: C = B D^2 B^T
    Matrix B = Matrix::Identity(n, n);
    Vector D = Vector::Ones(n);          // sqrt of eigenvalues
    Matrix invsqrtC = Matrix::Identity(n, n);

    // ── Bookkeeping ──────────────────────────────────────────────────────────
    SolverResult result;
    result.best_x = m;
    result.best_cost = problem.Evaluate(m);
    result.num_evaluations = 1;

    std::vector<Vector> xs(lam);
    std::vector<double> costs(lam);
    std::vector<int> idx(lam);

    for (int gen = 0; gen < opts_.max_iterations; ++gen) {
      if (result.num_evaluations >= opts_.max_evaluations) break;
      if (result.best_cost <= opts_.cost_target) break;

      // ── 1. Sample ────────────────────────────────────────────────────────
      for (int k = 0; k < lam; ++k) {
        Vector z = rng.RandNVector(n);
        xs[k] = m + sigma * (B * (D.asDiagonal() * z));
        costs[k] = problem.Evaluate(xs[k]);
        ++result.num_evaluations;
      }

      // ── 2. Sort ──────────────────────────────────────────────────────────
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(),
                [&](int a, int b) { return costs[a] < costs[b]; });

      if (costs[idx[0]] < result.best_cost) {
        result.best_cost = costs[idx[0]];
        result.best_x = xs[idx[0]];
      }
      result.cost_history.push_back(result.best_cost);

      if (opts_.verbose)
        printf("[CMA-ES] gen %4d  evals %6d  best %.6e  sigma %.4e\n",
               gen, result.num_evaluations, result.best_cost, sigma);

      // ── 3. Update mean ────────────────────────────────────────────────────
      Vector m_old = m;
      m = Vector::Zero(n);
      for (int i = 0; i < mu; ++i)
        m += w[i] * xs[idx[i]];

      Vector y_w = (m - m_old) / sigma;

      // ── 4. Update p_sigma ─────────────────────────────────────────────────
      p_sigma = (1.0 - c_sigma) * p_sigma
              + std::sqrt(c_sigma * (2.0 - c_sigma) * mueff) * (invsqrtC * y_w);

      // ── 5. Update sigma ───────────────────────────────────────────────────
      sigma *= std::exp((c_sigma / d_sigma) * (p_sigma.norm() / chi_n - 1.0));
      sigma = std::max(sigma, 1e-20);

      // ── 6. h_sigma indicator ─────────────────────────────────────────────
      double ps_norm_thresh =
          (1.4 + 2.0 / (n + 1.0)) * chi_n
          * std::sqrt(1.0 - std::pow(1.0 - c_sigma, 2.0 * (gen + 1)));
      const bool h_sigma = p_sigma.norm() < ps_norm_thresh;

      // ── 7. Update p_c ─────────────────────────────────────────────────────
      p_c = (1.0 - c_c) * p_c
          + (h_sigma ? 1.0 : 0.0)
            * std::sqrt(c_c * (2.0 - c_c) * mueff) * y_w;

      // ── 8. Update C ───────────────────────────────────────────────────────
      double c_hsig = (1.0 - h_sigma) * c_c * (2.0 - c_c);
      C = (1.0 - c_1 - c_mu) * C
        + c_1 * (p_c * p_c.transpose() + c_hsig * C);
      for (int i = 0; i < mu; ++i) {
        Vector yi = (xs[idx[i]] - m_old) / sigma;
        C += c_mu * w[i] * yi * yi.transpose();
      }

      // ── 9. Eigendecompose C (periodically) ────────────────────────────────
      if (gen % eigen_interval == 0) {
        // Enforce symmetry
        C = (C + C.transpose()) / 2.0;
        Eigen::SelfAdjointEigenSolver<Matrix> es(C);
        if (es.info() == Eigen::Success) {
          D = es.eigenvalues().cwiseSqrt().cwiseMax(1e-20);
          B = es.eigenvectors();
          invsqrtC = B * D.cwiseInverse().asDiagonal() * B.transpose();
        }
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.num_iterations = static_cast<int>(result.cost_history.size());
    return result;
  }

 private:
  CMAESOptions opts_;
};

}  // namespace global_optim
