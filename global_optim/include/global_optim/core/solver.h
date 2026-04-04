#pragma once

#include "global_optim/core/problem.h"
#include "global_optim/core/result.h"

#include <limits>
#include <string>

namespace global_optim {

// Base configuration shared by all solvers
struct SolverOptions {
  int max_iterations = 1000;
  int max_evaluations = 100000;
  double cost_target = -std::numeric_limits<double>::infinity();
  bool verbose = false;
  uint64_t seed = 42;
};

// Unified solver base class — CMA-ES, xNES, SVGD, MPPI all derive from this.
// All algorithms minimize Problem::Evaluate starting from an initial guess x0.
class Solver {
 public:
  virtual ~Solver() = default;

  virtual std::string Name() const = 0;

  // Solve the problem starting from x0.
  // x0 may be zero-vector or a warm-start from a previous solution.
  virtual SolverResult Solve(const Problem& problem, const Vector& x0) = 0;
};

}  // namespace global_optim
