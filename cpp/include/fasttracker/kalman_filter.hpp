#pragma once

#include <string>
#include <utility>
#include <vector>

#include "fasttracker/linear_algebra.hpp"

namespace fasttracker {

struct ProjectedState {
  std::vector<double> mean;
  Matrix covariance;
};

class KalmanFilter {
public:
  KalmanFilter();

  std::pair<std::vector<double>, Matrix> Initiate(
      const std::vector<double> &measurement) const;

  std::pair<std::vector<double>, Matrix> Predict(const std::vector<double> &mean,
                                                 const Matrix &covariance) const;

  std::pair<std::vector<std::vector<double>>, std::vector<Matrix>> MultiPredict(
      const std::vector<std::vector<double>> &means,
      const std::vector<Matrix> &covariances) const;

  ProjectedState Project(const std::vector<double> &mean,
                         const Matrix &covariance) const;

  std::pair<std::vector<double>, Matrix> Update(
      const std::vector<double> &mean, const Matrix &covariance,
      const std::vector<double> &measurement) const;

  std::vector<double> GatingDistance(
      const std::vector<double> &mean, const Matrix &covariance,
      const std::vector<std::vector<double>> &measurements,
      bool only_position = false, const std::string &metric = "maha") const;

  static double Chi2Inv95(int dof);

private:
  Matrix motion_mat_;
  Matrix update_mat_;
  double std_weight_position_;
  double std_weight_velocity_;
};

}  // namespace fasttracker

