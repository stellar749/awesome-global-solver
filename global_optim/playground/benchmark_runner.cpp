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

// Run a single (solver, problem) pair with population recording enabled.
// Saves convergence CSV + population CSV, then exits.
static void run_animate(const std::string& solver_name,
                        const std::string& problem_name,
                        const std::string& out_dir) {
  auto problems = make_problems();
  const ProblemEntry* pe = nullptr;
  for (const auto& p : problems)
    if (p.name == problem_name) { pe = &p; break; }

  if (!pe) {
    fprintf(stderr, "Unknown problem '%s'. Available:\n", problem_name.c_str());
    for (const auto& p : problems) fprintf(stderr, "  %s\n", p.name.c_str());
    exit(1);
  }

  auto make_solver = [&](uint64_t seed) -> SolverResult {
    if (solver_name == "cmaes") {
      CMAESOptions opts; opts.seed = seed;
      opts.max_iterations = 200; opts.max_evaluations = 50000;
      opts.sigma0 = 1.0; opts.record_population = true;
      return CMAESSolver(opts).Solve(*pe->problem, pe->x0_center);
    } else if (solver_name == "xnes") {
      XNESOptions opts; opts.seed = seed;
      opts.max_iterations = 200; opts.max_evaluations = 50000;
      opts.sigma0 = 1.0; opts.record_population = true;
      return XNESSolver(opts).Solve(*pe->problem, pe->x0_center);
    } else if (solver_name == "svgd") {
      if (!pe->problem->HasGradient()) {
        fprintf(stderr, "SVGD requires gradient; '%s' has none.\n", problem_name.c_str());
        exit(1);
      }
      SVGDOptions opts; opts.seed = seed;
      opts.max_iterations = 200; opts.num_particles = 50;
      opts.step_size = 0.01; opts.temperature = 0.1;
      opts.use_adagrad = false; opts.record_population = true;
      return SVGDSolver(opts).Solve(*pe->problem, pe->x0_center);
    } else if (solver_name == "mppi") {
      MPPIOptions opts; opts.seed = seed;
      opts.max_iterations = 200; opts.max_evaluations = 50000;
      opts.num_samples = 200; opts.noise_sigma = 0.5;
      opts.record_population = true;
      return MPPISolver(opts).Solve(*pe->problem, pe->x0_center);
    } else {
      fprintf(stderr, "Unknown solver '%s'. Use: cmaes xnes svgd mppi\n",
              solver_name.c_str());
      exit(1);
    }
  };

  fs::create_directories(out_dir);
  printf("Recording search process: %s on %s\n",
         solver_name.c_str(), problem_name.c_str());

  auto result = make_solver(42);
  printf("  best_cost=%.4e  evals=%d  generations=%d\n",
         result.best_cost, result.num_evaluations, result.num_iterations);

  // Save convergence CSV
  RunRecord rec;
  rec.solver_name  = solver_name;
  rec.problem_name = problem_name;
  rec.cost_history = result.cost_history;
  rec.eval_history = result.eval_history;
  std::string conv_path = out_dir + "/" + solver_name + "_" + problem_name + "_conv.csv";
  BenchmarkRunner::SaveConvergenceCSV(rec, conv_path);
  printf("  Convergence saved: %s\n", conv_path.c_str());

  // Save population history CSV
  std::string pop_path = out_dir + "/" + solver_name + "_" + problem_name + "_population.csv";
  BenchmarkRunner::SavePopulationCSV(solver_name, problem_name, result, pop_path);
  printf("  Population saved:  %s  (%zu frames)\n",
         pop_path.c_str(), result.population_history.size());

  printf("\nTo render animation:\n");
  printf("  python visualize.py --mode animation --solver %s --problem %s"
         " --results-dir %s --output-dir plots\n",
         solver_name.c_str(), problem_name.c_str(), out_dir.c_str());
}

int main(int argc, char** argv) {
  int num_seeds  = 51;
  std::string out_dir = "benchmark_results";
  std::string animate_solver, animate_problem;
  std::string filter_problem;   // if set, only run this problem (compare mode)

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--seeds" && i + 1 < argc)        num_seeds       = std::stoi(argv[++i]);
    if (arg == "--output-dir" && i + 1 < argc)   out_dir         = argv[++i];
    if (arg == "--problem" && i + 1 < argc)      filter_problem  = argv[++i];
    if (arg == "--animate" && i + 2 < argc) {
      animate_solver  = argv[++i];
      animate_problem = argv[++i];
    }
  }

  // Single-run animation mode
  if (!animate_solver.empty()) {
    run_animate(animate_solver, animate_problem, out_dir);
    return 0;
  }

  fs::create_directories(out_dir);

  BenchmarkRunner::Config cfg;
  cfg.num_seeds         = num_seeds;
  cfg.success_threshold = 1e-3;
  cfg.verbose           = false;
  BenchmarkRunner runner(cfg);

  auto problems = make_problems();

  // Solver name → factory function
  struct SolverEntry { std::string name; SolverFn fn; };
  std::vector<SolverEntry> solvers = {
    {"cmaes", run_cmaes},
    {"xnes",  run_xnes},
    {"svgd",  run_svgd},
    {"mppi",  run_mppi},
  };

  std::vector<RunRecord> all_records;
  std::vector<BenchmarkSummary> summaries;

  // When comparing a single problem, collect per-seed convergence for bands
  std::ofstream multi_conv_file;
  bool save_multi_conv = !filter_problem.empty();
  if (save_multi_conv) {
    std::string mc_path = out_dir + "/compare_" + filter_problem + "_conv.csv";
    multi_conv_file.open(mc_path);
    multi_conv_file << "solver,seed,eval,cost\n";
    printf("Compare mode: problem='%s', seeds=%d\n",
           filter_problem.c_str(), num_seeds);
  }

  for (const auto& prob : problems) {
    if (!filter_problem.empty() && prob.name != filter_problem) continue;

    printf("\n[%s  dim=%d]\n", prob.name.c_str(), prob.problem->Dimension());

    for (const auto& sol : solvers) {
      printf("  Running %-6s ... ", sol.name.c_str()); fflush(stdout);

      auto x0_fn = [&prob](uint64_t seed) -> Vector {
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

      for (auto& r : records) all_records.push_back(r);

      // Single-seed convergence curve (for animation overlay)
      std::string conv_path = out_dir + "/" + sol.name + "_" + prob.name + "_conv.csv";
      if (!records.empty())
        BenchmarkRunner::SaveConvergenceCSV(records[0], conv_path);

      // Multi-seed convergence (for compare dashboard bands)
      if (save_multi_conv) {
        for (const auto& rec : records) {
          for (size_t i = 0; i < rec.cost_history.size(); ++i) {
            int ev = i < rec.eval_history.size()
                     ? rec.eval_history[i] : static_cast<int>(i);
            multi_conv_file << sol.name << ',' << rec.seed << ','
                            << ev << ',' << rec.cost_history[i] << '\n';
          }
        }
      }
    }
  }

  // Print summary table
  BenchmarkRunner::PrintTable(summaries);

  // Save full per-run CSV
  BenchmarkRunner::SaveCSV(all_records, out_dir + "/benchmark_results.csv");
  printf("\nResults saved to %s/\n", out_dir.c_str());
  return 0;
}
