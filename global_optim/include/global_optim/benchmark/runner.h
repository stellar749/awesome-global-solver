#pragma once

#include "global_optim/core/problem.h"
#include "global_optim/core/result.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace global_optim {

// One row of the benchmark output table
struct RunRecord {
  std::string solver_name;
  std::string problem_name;
  int dim;
  uint64_t seed;
  Vector best_x;            // best solution found (for mode coverage analysis)
  double best_cost;
  int num_evaluations;
  int num_iterations;
  double elapsed_ms;
  std::vector<double> cost_history;
  std::vector<int> eval_history;
};

// Aggregated statistics over multiple seeds
struct BenchmarkSummary {
  std::string solver_name;
  std::string problem_name;
  int dim;
  int n_runs;

  double median_cost;
  double q25_cost;
  double q75_cost;
  double best_cost;          // best across all runs

  double success_rate;       // fraction with best_cost <= success_threshold
  double median_evaluations;

  double success_threshold;
};

// Solver factory: given a seed, returns a configured solver result on (problem, x0).
// Using a function rather than Solver& lets the runner vary the seed per run.
using SolverFn = std::function<SolverResult(const Problem&, const Vector&, uint64_t seed)>;

class BenchmarkRunner {
 public:
  struct Config {
    int num_seeds;
    uint64_t base_seed;
    double success_threshold;
    bool verbose;
    Config() : num_seeds(51), base_seed(0), success_threshold(1e-4), verbose(false) {}
  };

  explicit BenchmarkRunner(Config cfg = Config()) : cfg_(cfg) {}

  // Run one (solver, problem) pair over num_seeds independent seeds.
  // x0_fn maps seed → initial guess (default: zeros).
  std::vector<RunRecord> Run(
      const std::string& solver_name,
      SolverFn solver_fn,
      const std::string& problem_name,
      const Problem& problem,
      std::function<Vector(uint64_t)> x0_fn = nullptr) const {

    int d = problem.Dimension();
    std::vector<RunRecord> records;
    records.reserve(cfg_.num_seeds);

    for (int i = 0; i < cfg_.num_seeds; ++i) {
      uint64_t seed = cfg_.base_seed + static_cast<uint64_t>(i);
      Vector x0 = x0_fn ? x0_fn(seed) : Vector::Zero(d);

      SolverResult r = solver_fn(problem, x0, seed);

      RunRecord rec;
      rec.solver_name   = solver_name;
      rec.problem_name  = problem_name;
      rec.dim           = d;
      rec.seed          = seed;
      rec.best_x        = r.best_x;
      rec.best_cost     = r.best_cost;
      rec.num_evaluations = r.num_evaluations;
      rec.num_iterations  = r.num_iterations;
      rec.elapsed_ms    = r.elapsed_time_ms;
      rec.cost_history  = r.cost_history;
      rec.eval_history  = r.eval_history;
      records.push_back(std::move(rec));

      if (cfg_.verbose)
        printf("  [%s / %s] seed %3llu  best=%.4e  evals=%d\n",
               solver_name.c_str(), problem_name.c_str(),
               (unsigned long long)seed,
               records.back().best_cost, records.back().num_evaluations);
    }
    return records;
  }

  // Compute summary statistics from a set of run records.
  BenchmarkSummary Summarize(const std::vector<RunRecord>& records) const {
    if (records.empty()) return {};
    BenchmarkSummary s;
    s.solver_name  = records[0].solver_name;
    s.problem_name = records[0].problem_name;
    s.dim          = records[0].dim;
    s.n_runs       = static_cast<int>(records.size());
    s.success_threshold = cfg_.success_threshold;

    std::vector<double> costs;
    std::vector<double> evals;
    int successes = 0;
    for (const auto& r : records) {
      costs.push_back(r.best_cost);
      evals.push_back(static_cast<double>(r.num_evaluations));
      if (r.best_cost <= cfg_.success_threshold) ++successes;
    }

    std::sort(costs.begin(), costs.end());
    std::sort(evals.begin(), evals.end());

    auto percentile = [](const std::vector<double>& v, double p) {
      double idx = p * (v.size() - 1);
      int lo = static_cast<int>(idx);
      int hi = std::min(lo + 1, static_cast<int>(v.size()) - 1);
      return v[lo] + (idx - lo) * (v[hi] - v[lo]);
    };

    s.median_cost      = percentile(costs, 0.50);
    s.q25_cost         = percentile(costs, 0.25);
    s.q75_cost         = percentile(costs, 0.75);
    s.best_cost        = costs.front();
    s.success_rate     = static_cast<double>(successes) / s.n_runs;
    s.median_evaluations = percentile(evals, 0.50);
    return s;
  }

  // Save per-run records to CSV (one row per run, cost_history omitted).
  static void SaveCSV(const std::vector<RunRecord>& records,
                      const std::string& path) {
    std::ofstream f(path);
    f << "solver,problem,dim,seed,best_cost,num_evaluations,num_iterations,elapsed_ms\n";
    for (const auto& r : records) {
      f << r.solver_name << ','
        << r.problem_name << ','
        << r.dim << ','
        << r.seed << ','
        << r.best_cost << ','
        << r.num_evaluations << ','
        << r.num_iterations << ','
        << r.elapsed_ms << '\n';
    }
  }

  // Save convergence curves (eval_history + cost_history) for one record.
  // Format: eval,cost (one row per checkpoint).
  static void SaveConvergenceCSV(const RunRecord& rec, const std::string& path) {
    std::ofstream f(path);
    f << "solver,problem,seed,eval,cost\n";
    for (size_t i = 0; i < rec.cost_history.size(); ++i) {
      int ev = i < rec.eval_history.size() ? rec.eval_history[i] : static_cast<int>(i);
      f << rec.solver_name << ','
        << rec.problem_name << ','
        << rec.seed << ','
        << ev << ','
        << rec.cost_history[i] << '\n';
    }
  }

  // Save per-generation population positions from a SolverResult to CSV.
  // Format: gen,eval,particle_id,x0,x1,...,x{d-1}
  // Requires SolverOptions::record_population = true when solving.
  static void SavePopulationCSV(const std::string& solver_name,
                                 const std::string& problem_name,
                                 const SolverResult& result,
                                 const std::string& path) {
    if (result.population_history.empty()) return;
    std::ofstream f(path);
    const int d = static_cast<int>(result.population_history[0].cols());
    f << "solver,problem,gen,eval,particle_id";
    for (int j = 0; j < d; ++j) f << ",x" << j;
    f << '\n';
    for (size_t g = 0; g < result.population_history.size(); ++g) {
      int ev = g < result.population_eval_history.size()
               ? result.population_eval_history[g] : static_cast<int>(g);
      const Matrix& pop = result.population_history[g];
      for (int k = 0; k < static_cast<int>(pop.rows()); ++k) {
        f << solver_name << ',' << problem_name << ','
          << g << ',' << ev << ',' << k;
        for (int j = 0; j < d; ++j) f << ',' << pop(k, j);
        f << '\n';
      }
    }
  }

  // Compute ECDF of best_cost values from a set of run records.
  // Returns (sorted_costs, cumulative_fractions) as parallel vectors.
  // Plot: ax.step(costs, fractions, where="post") gives the ECDF curve.
  static std::pair<std::vector<double>, std::vector<double>>
  ComputeECDF(const std::vector<RunRecord>& records) {
    std::vector<double> costs;
    costs.reserve(records.size());
    for (const auto& r : records) costs.push_back(r.best_cost);
    std::sort(costs.begin(), costs.end());
    const int n = static_cast<int>(costs.size());
    std::vector<double> cdf(n);
    for (int i = 0; i < n; ++i) cdf[i] = static_cast<double>(i + 1) / n;
    return {costs, cdf};
  }

  // Count distinct modes found across runs.
  // Two best_x solutions belong to the same mode if their Euclidean distance
  // is less than cluster_radius. Uses greedy nearest-neighbour clustering.
  static int CountModes(const std::vector<RunRecord>& records,
                        double cluster_radius) {
    std::vector<Vector> centers;
    for (const auto& r : records) {
      if (r.best_x.size() == 0) continue;
      bool new_mode = true;
      for (const auto& c : centers) {
        if ((r.best_x - c).norm() < cluster_radius) {
          new_mode = false;
          break;
        }
      }
      if (new_mode) centers.push_back(r.best_x);
    }
    return static_cast<int>(centers.size());
  }

  // Print a compact summary table to stdout.
  static void PrintTable(const std::vector<BenchmarkSummary>& summaries) {
    printf("\n%-10s %-22s %4s  %10s  %10s  %10s  %8s  %8s\n",
           "Solver", "Problem", "Dim",
           "Median", "Q25", "Q75", "Success%", "Med.Evals");
    printf("%s\n", std::string(90, '-').c_str());
    for (const auto& s : summaries) {
      printf("%-10s %-22s %4d  %10.3e  %10.3e  %10.3e  %7.1f%%  %8.0f\n",
             s.solver_name.c_str(), s.problem_name.c_str(), s.dim,
             s.median_cost, s.q25_cost, s.q75_cost,
             100.0 * s.success_rate, s.median_evaluations);
    }
  }

 private:
  Config cfg_;
};


}  // namespace global_optim
