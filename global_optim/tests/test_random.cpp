#include "global_optim/core/random.h"

#include <gtest/gtest.h>
#include <cmath>

using namespace global_optim;

TEST(RandomEngine, Reproducible) {
  RandomEngine r1(123), r2(123);
  for (int i = 0; i < 100; ++i)
    EXPECT_DOUBLE_EQ(r1.RandN(), r2.RandN());
}

TEST(RandomEngine, DifferentSeeds) {
  RandomEngine r1(1), r2(2);
  bool any_different = false;
  for (int i = 0; i < 20; ++i)
    if (r1.RandN() != r2.RandN()) { any_different = true; break; }
  EXPECT_TRUE(any_different);
}

TEST(RandomEngine, RandNVectorLength) {
  RandomEngine rng(42);
  auto v = rng.RandNVector(10);
  EXPECT_EQ(v.size(), 10);
}

TEST(RandomEngine, RandNMatrixShape) {
  RandomEngine rng(42);
  auto m = rng.RandNMatrix(5, 3);
  EXPECT_EQ(m.rows(), 5);
  EXPECT_EQ(m.cols(), 3);
}

TEST(RandomEngine, RandNApproximatelyZeroMean) {
  RandomEngine rng(0);
  int n = 10000;
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += rng.RandN();
  EXPECT_NEAR(sum / n, 0.0, 0.05);  // 5-sigma for this n
}

TEST(RandomEngine, RandUInRange) {
  RandomEngine rng(7);
  for (int i = 0; i < 1000; ++i) {
    double u = rng.RandU();
    EXPECT_GE(u, 0.0);
    EXPECT_LT(u, 1.0);
  }
}

TEST(RandomEngine, SampleIsotropicShape) {
  RandomEngine rng(42);
  Vector mean = Vector::Zero(4);
  auto samples = rng.SampleIsotropic(mean, 1.0, 100);
  EXPECT_EQ(samples.rows(), 100);
  EXPECT_EQ(samples.cols(), 4);
}

TEST(RandomEngine, SampleMVNMean) {
  // Sample many points from N(mu, I) and check empirical mean
  RandomEngine rng(42);
  int d = 3, n = 5000;
  Vector mu(d);
  mu << 1.0, -2.0, 3.0;
  Matrix L = Matrix::Identity(d, d);

  Matrix samples(n, d);
  for (int i = 0; i < n; ++i)
    samples.row(i) = rng.SampleMVN(mu, L);

  Vector empirical_mean = samples.colwise().mean();
  for (int j = 0; j < d; ++j)
    EXPECT_NEAR(empirical_mean[j], mu[j], 0.1);
}
