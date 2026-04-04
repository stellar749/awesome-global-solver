#pragma once

#include "global_optim/core/types.h"

namespace global_optim {

// Unified optimization problem interface.
// All problems (static optimization, trajectory optimization) are solved through
// this interface: x -> cost (minimization).
class Problem {
 public:
  virtual ~Problem() = default;

  // Objective function: x -> cost (minimize)
  virtual double Evaluate(const Vector& x) const = 0;

  // Problem dimension
  virtual int Dimension() const = 0;

  // Search space bounds (optional; empty vector = unbounded)
  virtual Vector LowerBound() const { return {}; }
  virtual Vector UpperBound() const { return {}; }

  // Gradient (optional, required by SVGD)
  virtual Vector Gradient(const Vector& x) const {
    throw std::runtime_error("Gradient not implemented for this problem");
  }

  virtual bool HasGradient() const { return false; }
};

}  // namespace global_optim
