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

struct XNESOptions : SolverOptions {
  int lambda = -1;          // population size; default: 4 + floor(3*ln(n))
  double eta_mu = 1.0;      // mean learning rate
  double eta_sigma = -1;    // step-size learning rate; default: (3+ln(n))/(5*sqrt(n))
  double eta_B = -1;        // shape learning rate; default: same as eta_sigma
  double sigma0 = 1.0;      // initial step size
};

// Exponential Natural Evolution Strategies (Glasmachers et al. 2010).
// Maintains search distribution N(mu, sigma^2 * B * B^T) and updates
// all parameters via natural gradient in the exponential parameterization.
class XNESSolver : public Solver {
 public:
  explicit XNESSolver(XNESOptions opts = {}) : opts_(opts) {}

  std::string Name() const override { return "xNES"; }

  SolverResult Solve(const Problem& problem, const Vector& x0) override {
    auto t0 = std::chrono::steady_clock::now();
    const int n = problem.Dimension();
    RandomEngine rng(opts_.seed);

    // ── Hyper-parameters ─────────────────────────────────────────────────────
    const int lam = opts_.lambda > 0 ? opts_.lambda
                                     : 4 + static_cast<int>(3.0 * std::log(n));
    const double eta_mu = opts_.eta_mu;
    const double eta_sigma = opts_.eta_sigma > 0 ? opts_.eta_sigma
        : (3.0 + std::log(n)) / (5.0 * n * std::sqrt(static_cast<double>(n)));
    const double eta_B = opts_.eta_B > 0 ? opts_.eta_B : eta_sigma;

    // Rank-based utility function (fitness shaping)
    // u_k = max(0, ln(lam/2+1) - ln(k+1)), k=0..lam-1 (0 = best)
    // normalize, then center
    std::vector<double> u(lam);
    double u_sum = 0;
    for (int k = 0; k < lam; ++k) {
      u[k] = std::max(0.0, std::log(lam / 2.0 + 1.0) - std::log(k + 1.0));
      u_sum += u[k];
    }
    for (double& uk : u) {
      uk = uk / u_sum - 1.0 / lam;
    }

    // ── State ─────────────────────────────────────────────────────────────────
    Vector mu = x0;
    double sigma = opts_.sigma0;
    Matrix B = Matrix::Identity(n, n);  // shape: x_i = mu + sigma * B * z_i

    // ── Bookkeeping ───────────────────────────────────────────────────────────
    SolverResult result;
    result.best_x = mu;
    result.best_cost = problem.Evaluate(mu);
    result.num_evaluations = 1;

    std::vector<Vector> zs(lam);
    std::vector<double> costs(lam);
    std::vector<int> idx(lam);

    for (int gen = 0; gen < opts_.max_iterations; ++gen) {
      if (result.num_evaluations >= opts_.max_evaluations) break;
      if (result.best_cost <= opts_.cost_target) break;

      // ── 1. Sample ────────────────────────────────────────────────────────
      for (int k = 0; k < lam; ++k) {
        zs[k] = rng.RandNVector(n);
        costs[k] = problem.Evaluate(mu + sigma * (B * zs[k]));
        ++result.num_evaluations;
      }

      // ── 2. Sort by fitness (ascending = minimize) ─────────────────────────
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(),
                [&](int a, int b) { return costs[a] < costs[b]; });

      // Track best
      {
        Vector best_x = mu + sigma * (B * zs[idx[0]]);
        if (costs[idx[0]] < result.best_cost) {
          result.best_cost = costs[idx[0]];
          result.best_x = best_x;
        }
      }
      result.cost_history.push_back(result.best_cost);
      result.eval_history.push_back(result.num_evaluations);

      if (opts_.record_population) {
        Matrix pop(lam, n);
        for (int k = 0; k < lam; ++k)
          pop.row(k) = (mu + sigma * (B * zs[k])).transpose();
        result.population_history.push_back(pop);
        result.population_eval_history.push_back(result.num_evaluations);
      }

      if (opts_.verbose)
        printf("[xNES] gen %4d  evals %6d  best %.6e  sigma %.4e\n",
               gen, result.num_evaluations, result.best_cost, sigma);

      // ── 3. Natural gradients ──────────────────────────────────────────────
      // G_delta = sum_k u_{rank(k)} * z_k
      // G_M     = sum_k u_{rank(k)} * (z_k z_k^T - I)
      Vector G_delta = Vector::Zero(n);
      Matrix G_M = Matrix::Zero(n, n);
      for (int k = 0; k < lam; ++k) {
        double uk = u[k];               // u[0] = utility for rank-1 individual
        const Vector& z = zs[idx[k]];  // idx sorted best→worst
        G_delta += uk * z;
        G_M += uk * (z * z.transpose() - Matrix::Identity(n, n));
      }

      // ── 4. Decompose G_M into G_sigma and G_B ─────────────────────────────
      double G_sigma = G_M.trace() / n;
      Matrix G_B = G_M - G_sigma * Matrix::Identity(n, n);

      // ── 5. Update parameters ──────────────────────────────────────────────
      mu += eta_mu * sigma * (B * G_delta);
      sigma *= std::exp(eta_sigma / 2.0 * G_sigma);
      sigma = std::max(sigma, 1e-20);

      // B *= expm(eta_B/2 * G_B)
      // G_B is symmetric → use eigendecomposition for matrix exponential
      Matrix A = (eta_B / 2.0) * G_B;
      A = (A + A.transpose()) / 2.0;  // enforce symmetry
      Eigen::SelfAdjointEigenSolver<Matrix> es(A);
      if (es.info() == Eigen::Success) {
        Matrix expmA = es.eigenvectors()
                     * es.eigenvalues().array().exp().matrix().asDiagonal()
                     * es.eigenvectors().transpose();
        B = B * expmA;
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.num_iterations = static_cast<int>(result.cost_history.size());
    return result;
  }

 private:
  XNESOptions opts_;
};

}  // namespace global_optim
