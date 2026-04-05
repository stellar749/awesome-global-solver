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

struct MPPIOptions : SolverOptions {
  int num_samples = 1024;    // K: number of perturbed rollouts per iteration
  double noise_sigma = 0.5;  // isotropic perturbation std dev
  double temperature = 1.0;  // λ: inverse temperature for IS weights
};

// Model Predictive Path Integral control as a generic black-box optimizer.
//
// Treats the optimization variable x (e.g., a flattened control sequence) as
// the current solution and refines it iteratively via importance-sampled
// perturbations:
//   1. Sample K perturbations: ε_k ~ N(0, sigma^2 I)
//   2. Evaluate S_k = f(x + ε_k)
//   3. Compute weights: w_k ∝ exp(-(S_k - min_S) / λ)   (softmax)
//   4. Update: x ← Σ_k w_k (x + ε_k)
//
// This is equivalent to one step of IS-weighted zero-order policy gradient.
// When wrapped around a ControlSequenceProblem, it recovers standard MPPI.
class MPPISolver : public Solver {
 public:
  explicit MPPISolver(MPPIOptions opts = {}) : opts_(opts) {}

  std::string Name() const override { return "MPPI"; }

  SolverResult Solve(const Problem& problem, const Vector& x0) override {
    auto t0 = std::chrono::steady_clock::now();
    const int d = problem.Dimension();
    RandomEngine rng(opts_.seed);

    const int K = opts_.num_samples;
    const double sigma = opts_.noise_sigma;
    const double inv_temp = opts_.temperature;

    // ── State ─────────────────────────────────────────────────────────────────
    Vector x = x0;

    SolverResult result;
    result.best_x = x;
    result.best_cost = problem.Evaluate(x);
    result.num_evaluations = 1;

    std::vector<Vector> eps(K);
    std::vector<double> costs(K);

    for (int iter = 0; iter < opts_.max_iterations; ++iter) {
      if (result.num_evaluations >= opts_.max_evaluations) break;
      if (result.best_cost <= opts_.cost_target) break;

      // ── 1. Sample K perturbations and evaluate ────────────────────────────
      double min_cost = std::numeric_limits<double>::infinity();
      for (int k = 0; k < K; ++k) {
        eps[k] = sigma * rng.RandNVector(d);
        costs[k] = problem.Evaluate(x + eps[k]);
        ++result.num_evaluations;
        if (costs[k] < min_cost) min_cost = costs[k];
      }

      // ── 2. Compute IS weights (numerically stable softmax) ─────────────────
      std::vector<double> w(K);
      double w_sum = 0;
      for (int k = 0; k < K; ++k) {
        w[k] = std::exp(-(costs[k] - min_cost) / inv_temp);
        w_sum += w[k];
      }

      // ── 3. Weighted update ─────────────────────────────────────────────────
      Vector x_new = Vector::Zero(d);
      for (int k = 0; k < K; ++k) {
        x_new += (w[k] / w_sum) * (x + eps[k]);
      }
      x = x_new;

      // ── 4. Track best ─────────────────────────────────────────────────────
      double cur_cost = problem.Evaluate(x);
      ++result.num_evaluations;
      if (cur_cost < result.best_cost) {
        result.best_cost = cur_cost;
        result.best_x = x;
      }
      result.cost_history.push_back(result.best_cost);
      result.eval_history.push_back(result.num_evaluations);

      if (opts_.record_population) {
        Matrix pop(K, d);
        for (int k = 0; k < K; ++k) pop.row(k) = (x + eps[k]).transpose();
        result.population_history.push_back(pop);
        result.population_eval_history.push_back(result.num_evaluations);
      }

      if (opts_.verbose)
        printf("[MPPI] iter %4d  evals %6d  best %.6e\n",
               iter, result.num_evaluations, result.best_cost);
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.num_iterations = static_cast<int>(result.cost_history.size());
    return result;
  }

 private:
  MPPIOptions opts_;
};

}  // namespace global_optim
