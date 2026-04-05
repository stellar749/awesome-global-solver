// benchmark_runner.cpp — Phase 3 benchmark executable
//
// Runs CMA-ES, xNES, SVGD on all L2 multimodal functions,
// collects results across 51 seeds, prints summary table,
// and saves CSVs to ./benchmark_results/ for Python visualization.
//
// Usage:
//   ./benchmark_runner [--seeds N] [--output-dir DIR]

#include "global_optim/benchmark/runner.h"
#include "global_optim/problems/benchmark_functions.h"
#include "global_optim/solvers/cmaes.h"
#include "global_optim/solvers/mppi.h"
#include "global_optim/solvers/svgd.h"
#include "global_optim/solvers/xnes.h"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace global_optim;
namespace fs = std::filesystem;

// ── Solver factories ──────────────────────────────────────────────────────────
// Each factory returns a SolverResult given (problem, x0, seed).

SolverResult run_cmaes(const Problem& p, const Vector& x0, uint64_t seed) {
  CMAESOptions opts;
  opts.seed = seed;
  opts.max_iterations = 1000;
  opts.max_evaluations = 200000;
  opts.sigma0 = 1.0;
  return CMAESSolver(opts).Solve(p, x0);
}

SolverResult run_xnes(const Problem& p, const Vector& x0, uint64_t seed) {
  XNESOptions opts;
  opts.seed = seed;
  opts.max_iterations = 1000;
  opts.max_evaluations = 200000;
  opts.sigma0 = 1.0;
  return XNESSolver(opts).Solve(p, x0);
}

SolverResult run_svgd(const Problem& p, const Vector& x0, uint64_t seed) {
  if (!p.HasGradient()) {
    // Return a trivial result for problems without gradient
    SolverResult r;
    r.best_x   = x0;
    r.best_cost = p.Evaluate(x0);
    r.num_evaluations = 1;
    r.num_iterations  = 0;
    r.elapsed_time_ms = 0;
    return r;
  }
  SVGDOptions opts;
  opts.seed = seed;
  opts.max_iterations  = 500;
  opts.max_evaluations = 200000;
  opts.num_particles   = 50;
  opts.step_size       = 0.01;
  opts.temperature     = 0.1;
  opts.use_adagrad     = false;
  return SVGDSolver(opts).Solve(p, x0);
}

SolverResult run_mppi(const Problem& p, const Vector& x0, uint64_t seed) {
  MPPIOptions opts;
  opts.seed = seed;
  opts.max_iterations  = 300;
  opts.max_evaluations = 200000;
  opts.num_samples     = 512;
  opts.noise_sigma     = 0.5;
  opts.temperature     = 1.0;
  return MPPISolver(opts).Solve(p, x0);
}

// ── Problem registry ──────────────────────────────────────────────────────────

struct ProblemEntry {
  std::string name;
  std::shared_ptr<Problem> problem;
  Vector x0_center;   // solver starts from x0_center + small noise
  bool needs_gradient;
};

std::vector<ProblemEntry> make_problems() {
  std::vector<ProblemEntry> ps;
  auto add = [&](std::string name, std::shared_ptr<Problem> p,
                 Vector x0, bool needs_grad = false) {
    ps.push_back({name, p, x0, needs_grad});
  };

  // L2 multimodal functions
  add("rastrigin_2d",  std::make_shared<RastriginProblem>(2),
      Vector::Constant(2, 2.0));
  add("rastrigin_10d", std::make_shared<RastriginProblem>(10),
      Vector::Constant(10, 2.0));
  add("ackley_2d",     std::make_shared<AckleyProblem>(2),
      Vector::Constant(2, 15.0));
  add("ackley_10d",    std::make_shared<AckleyProblem>(10),
      Vector::Constant(10, 10.0));
  add("schwefel_2d",   std::make_shared<SchwefelProblem>(2),
      Vector::Constant(2, 200.0));
  add("dbl_rosen_2d",  std::make_shared<DoubleRosenbrockProblem>(2),
      Vector::Zero(2));
  add("gauss_mix_2d",  std::make_shared<GaussianMixtureProblem>(2, 3, 4.0),
      Vector::Zero(2), /*needs_gradient=*/true);
  add("griewank_2d",   std::make_shared<GriewankProblem>(2),
      Vector::Constant(2, 200.0));
  add("griewank_10d",  std::make_shared<GriewankProblem>(10),
      Vector::Constant(10, 200.0));
  add("random_basin_2d", std::make_shared<RandomBasinProblem>(2),
      Vector::Constant(2, 0.0));
  add("random_basin_4d", std::make_shared<RandomBasinProblem>(4),
      Vector::Constant(4, 0.0));

  return ps;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  int num_seeds  = 51;
  std::string out_dir = "benchmark_results";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--seeds" && i + 1 < argc)   num_seeds  = std::stoi(argv[++i]);
    if (arg == "--output-dir" && i + 1 < argc) out_dir = argv[++i];
  }

  fs::create_directories(out_dir);

  BenchmarkRunner::Config cfg;
  cfg.num_seeds         = num_seeds;
  cfg.success_threshold = 1e-3;
  cfg.verbose           = false;
  BenchmarkRunner runner(cfg);

  auto problems = make_problems();

  // Solver name → factory function
  struct SolverEntry { std::string name; SolverFn fn; bool needs_grad; };
  std::vector<SolverEntry> solvers = {
    {"cmaes", run_cmaes, false},
    {"xnes",  run_xnes,  false},
    {"svgd",  run_svgd,  true},
    {"mppi",  run_mppi,  false},
  };

  std::vector<RunRecord> all_records;
  std::vector<BenchmarkSummary> summaries;

  for (const auto& prob : problems) {
    printf("\n[%s  dim=%d]\n", prob.name.c_str(), prob.problem->Dimension());

    for (const auto& sol : solvers) {
      // SVGD skips problems without gradient (returns trivial result)
      // Other solvers run on all problems
      printf("  Running %-6s ... ", sol.name.c_str()); fflush(stdout);

      auto x0_fn = [&prob](uint64_t seed) -> Vector {
        // Start from x0_center + small Gaussian noise
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> nd(0.0, 0.5);
        Vector x0 = prob.x0_center;
        for (int i = 0; i < x0.size(); ++i) x0[i] += nd(rng);
        return x0;
      };

      auto records = runner.Run(sol.name, sol.fn, prob.name, *prob.problem, x0_fn);
      auto summary = runner.Summarize(records);
      summaries.push_back(summary);

      printf("median=%.3e  success=%.0f%%\n",
             summary.median_cost, 100.0 * summary.success_rate);

      // Append to global list
      for (auto& r : records) all_records.push_back(r);

      // Save per-solver per-problem convergence curve (median seed)
      std::string conv_path = out_dir + "/" + sol.name + "_" + prob.name + "_conv.csv";
      // Save the first run's convergence curve as example
      if (!records.empty())
        BenchmarkRunner::SaveConvergenceCSV(records[0], conv_path);
    }
  }

  // Print summary table
  BenchmarkRunner::PrintTable(summaries);

  // Save full per-run CSV
  BenchmarkRunner::SaveCSV(all_records, out_dir + "/benchmark_results.csv");
  printf("\nResults saved to %s/\n", out_dir.c_str());
  return 0;
}
