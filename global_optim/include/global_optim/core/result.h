#pragma once

#include "global_optim/core/types.h"

namespace global_optim {

struct SolverResult {
  Vector best_x;               // Best solution found
  double best_cost;            // Best cost achieved
  int num_evaluations = 0;     // Total function evaluations
  int num_iterations = 0;      // Total iterations completed
  double elapsed_time_ms = 0;  // Wall-clock time

  // Best cost per iteration (for convergence plots)
  std::vector<double> cost_history;
};

// Trajectory extracted from a ControlSequenceProblem result
struct TrajectoryInfo {
  Matrix controls;  // T x control_dim
  Matrix states;    // (T+1) x state_dim
};

}  // namespace global_optim
