#pragma once

#include "global_optim/core/types.h"

#include <random>
#include <cassert>

namespace global_optim {

// Random number utilities used by all stochastic solvers.
// Each solver owns a RandomEngine instance seeded from SolverOptions::seed.
class RandomEngine {
 public:
  explicit RandomEngine(uint64_t seed = 42) : rng_(seed) {}

  void Seed(uint64_t seed) { rng_.seed(seed); }

  // Scalar standard normal N(0,1)
  double RandN() { return normal_(rng_); }

  // Scalar uniform U(0,1)
  double RandU() { return uniform_(rng_); }

  // Uniform integer in [lo, hi]
  int RandInt(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
  }

  // Vector of iid N(0,1) samples, length n
  Vector RandNVector(int n) {
    Vector v(n);
    for (int i = 0; i < n; ++i) v[i] = normal_(rng_);
    return v;
  }

  // Matrix of iid N(0,1) samples, shape rows x cols
  Matrix RandNMatrix(int rows, int cols) {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        m(i, j) = normal_(rng_);
    return m;
  }

  // Sample from multivariate N(mean, cov) via Cholesky of cov.
  // chol_L must be the lower-triangular Cholesky factor of the covariance.
  Vector SampleMVN(const Vector& mean, const Matrix& chol_L) {
    assert(mean.size() == chol_L.rows());
    return mean + chol_L * RandNVector(mean.size());
  }

  // Sample n points from N(mean, sigma^2 * I)
  Matrix SampleIsotropic(const Vector& mean, double sigma, int n) {
    int d = mean.size();
    Matrix samples(n, d);
    for (int i = 0; i < n; ++i)
      samples.row(i) = mean + sigma * RandNVector(d);
    return samples;
  }

  std::mt19937_64& RNG() { return rng_; }

 private:
  std::mt19937_64 rng_;
  std::normal_distribution<double> normal_{0.0, 1.0};
  std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

}  // namespace global_optim
