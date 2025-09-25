#include "fasttracker/kalman_filter.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace fasttracker {

namespace {
constexpr double kStdWeightPosition = 1.0 / 20.0;
constexpr double kStdWeightVelocity = 1.0 / 160.0;

const double kChi2Inv95Table[] = {
    0.0,    3.8415, 5.9915, 7.8147, 9.4877,
    11.070, 12.592, 14.067, 15.507, 16.919};

Matrix BuildMotionMatrix() {
  Matrix motion(8, 8);
  for (int i = 0; i < 8; ++i) {
    motion(i, i) = 1.0;
  }
  for (int i = 0; i < 4; ++i) {
    motion(i, i + 4) = 1.0;
  }
  return motion;
}

Matrix BuildUpdateMatrix() {
  Matrix update(4, 8);
  for (int i = 0; i < 4; ++i) {
    update(i, i) = 1.0;
  }
  return update;
}

}  // namespace

KalmanFilter::KalmanFilter()
    : motion_mat_(BuildMotionMatrix()), update_mat_(BuildUpdateMatrix()),
      std_weight_position_(kStdWeightPosition),
      std_weight_velocity_(kStdWeightVelocity) {}

std::pair<std::vector<double>, Matrix> KalmanFilter::Initiate(
    const std::vector<double> &measurement) const {
  std::vector<double> mean_pos = measurement;
  std::vector<double> mean_vel(mean_pos.size(), 0.0);
  std::vector<double> mean(mean_pos.size() * 2, 0.0);
  for (std::size_t i = 0; i < mean_pos.size(); ++i) {
    mean[i] = mean_pos[i];
    mean[i + mean_pos.size()] = mean_vel[i];
  }

  std::vector<double> std = {
      2 * std_weight_position_ * measurement[3],
      2 * std_weight_position_ * measurement[3],
      1e-2,
      2 * std_weight_position_ * measurement[3],
      10 * std_weight_velocity_ * measurement[3],
      10 * std_weight_velocity_ * measurement[3],
      1e-5,
      10 * std_weight_velocity_ * measurement[3]};
  std::vector<double> variance(std.size());
  for (std::size_t i = 0; i < std.size(); ++i) {
    variance[i] = std[i] * std[i];
  }
  Matrix covariance = Diagonal(variance);
  return {mean, covariance};
}

std::pair<std::vector<double>, Matrix> KalmanFilter::Predict(
    const std::vector<double> &mean, const Matrix &covariance) const {
  std::vector<double> std_pos = {
      std_weight_position_ * mean[3],
      std_weight_position_ * mean[3],
      1e-2,
      std_weight_position_ * mean[3]};
  std::vector<double> std_vel = {
      std_weight_velocity_ * mean[3],
      std_weight_velocity_ * mean[3],
      1e-5,
      std_weight_velocity_ * mean[3]};

  std::vector<double> concat;
  concat.reserve(std_pos.size() + std_vel.size());
  concat.insert(concat.end(), std_pos.begin(), std_pos.end());
  concat.insert(concat.end(), std_vel.begin(), std_vel.end());

  for (double &v : concat) {
    v = v * v;
  }
  Matrix motion_cov = Diagonal(concat);

  Matrix motion_t = Transpose(motion_mat_);
  std::vector<double> new_mean = VecMat(mean, motion_t);
  Matrix new_cov = MatMul(MatMul(motion_mat_, covariance), motion_t);
  new_cov = Add(new_cov, motion_cov);

  return {new_mean, new_cov};
}

std::pair<std::vector<std::vector<double>>, std::vector<Matrix>>
KalmanFilter::MultiPredict(const std::vector<std::vector<double>> &means,
                           const std::vector<Matrix> &covariances) const {
  std::vector<std::vector<double>> new_means;
  std::vector<Matrix> new_covs;
  new_means.reserve(means.size());
  new_covs.reserve(covariances.size());
  for (std::size_t i = 0; i < means.size(); ++i) {
    auto result = Predict(means[i], covariances[i]);
    new_means.push_back(result.first);
    new_covs.push_back(result.second);
  }
  return {new_means, new_covs};
}

ProjectedState KalmanFilter::Project(const std::vector<double> &mean,
                                     const Matrix &covariance) const {
  std::vector<double> std = {
      std_weight_position_ * mean[3],
      std_weight_position_ * mean[3],
      1e-1,
      std_weight_position_ * mean[3]};
  std::vector<double> variance(std.size());
  for (std::size_t i = 0; i < std.size(); ++i) {
    variance[i] = std[i] * std[i];
  }
  Matrix innovation_cov = Diagonal(variance);

  std::vector<double> projected_mean = MatVec(update_mat_, mean);
  Matrix projected_cov = MatMul(MatMul(update_mat_, covariance),
                                Transpose(update_mat_));
  projected_cov = Add(projected_cov, innovation_cov);

  return {projected_mean, projected_cov};
}

std::pair<std::vector<double>, Matrix> KalmanFilter::Update(
    const std::vector<double> &mean, const Matrix &covariance,
    const std::vector<double> &measurement) const {
  ProjectedState proj = Project(mean, covariance);

  Matrix gain_numerator = MatMul(covariance, Transpose(update_mat_));
  Matrix innovation_cov_inv = Inverse(proj.covariance);
  Matrix kalman_gain = MatMul(gain_numerator, innovation_cov_inv);

  std::vector<double> innovation(measurement.size());
  for (std::size_t i = 0; i < measurement.size(); ++i) {
    innovation[i] = measurement[i] - proj.mean[i];
  }

  std::vector<double> mean_correction = MatVec(kalman_gain, innovation);
  std::vector<double> new_mean = Add(mean, mean_correction);

  Matrix temp = MatMul(kalman_gain, proj.covariance);
  Matrix reduction = MatMul(temp, Transpose(kalman_gain));
  Matrix new_covariance = Subtract(covariance, reduction);

  return {new_mean, new_covariance};
}

std::vector<double> KalmanFilter::GatingDistance(
    const std::vector<double> &mean, const Matrix &covariance,
    const std::vector<std::vector<double>> &measurements, bool only_position,
    const std::string &metric) const {
  ProjectedState proj = Project(mean, covariance);
  std::vector<double> local_mean = proj.mean;
  Matrix local_cov = proj.covariance;

  if (only_position) {
    local_mean = {local_mean[0], local_mean[1]};
    Matrix reduced(2, 2);
    for (int r = 0; r < 2; ++r) {
      for (int c = 0; c < 2; ++c) {
        reduced(r, c) = local_cov(r, c);
      }
    }
    local_cov = reduced;
  }

  Matrix inv_cov = Inverse(local_cov);
  std::vector<double> result(measurements.size(), 0.0);
  for (std::size_t i = 0; i < measurements.size(); ++i) {
    std::vector<double> diff = measurements[i];
    diff = Subtract(diff, local_mean);
    if (only_position && diff.size() > 2) {
      diff.resize(2);
    }

    if (metric == "maha" || metric.empty()) {
      std::vector<double> intermediate = MatVec(inv_cov, diff);
      result[i] = Dot(diff, intermediate);
    } else if (metric == "gaussian") {
      result[i] = Dot(diff, diff);
    } else {
      throw std::invalid_argument("Unknown gating metric");
    }
  }
  return result;
}

double KalmanFilter::Chi2Inv95(int dof) {
  if (dof < 1 || dof >= static_cast<int>(sizeof(kChi2Inv95Table) /
                                         sizeof(kChi2Inv95Table[0]))) {
    throw std::out_of_range("Chi2Inv95 table index out of range");
  }
  return kChi2Inv95Table[dof];
}

}  // namespace fasttracker

