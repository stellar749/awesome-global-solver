#pragma once

#include "global_optim/core/types.h"

namespace global_optim {

struct SolverResult {
  Vector best_x;               // Best solution found
  double best_cost;            // Best cost achieved
  int num_evaluations = 0;     // Total function evaluations
  int num_iterations = 0;      // Total iterations completed
  double elapsed_time_ms = 0;  // Wall-clock time

  // Convergence history — parallel arrays, one entry per recorded iteration.
  // Use eval_history as x-axis for fair cross-algorithm comparison.
  std::vector<double> cost_history;  // best cost at each checkpoint
  std::vector<int> eval_history;     // cumulative evaluations at each checkpoint

  // Per-generation population snapshots (only populated when
  // SolverOptions::record_population = true).
  // population_history[g] is a (population_size x dim) matrix of sample positions.
  // population_eval_history[g] is the cumulative eval count at generation g.
  std::vector<Matrix> population_history;
  std::vector<int>    population_eval_history;
};

// Trajectory extracted from a ControlSequenceProblem result
struct TrajectoryInfo {
  Matrix controls;  // T x control_dim
  Matrix states;    // (T+1) x state_dim
};

}  // namespace global_optim
