#pragma once

#include <utility>
#include <vector>

#include "fasttracker/linear_algebra.hpp"
#include "fasttracker/strack.hpp"

namespace fasttracker {

struct AssignmentResult {
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_rows;
  std::vector<int> unmatched_cols;
};

Matrix IouDistance(const std::vector<std::shared_ptr<STrack>> &a,
                   const std::vector<std::shared_ptr<STrack>> &b);
Matrix IouDistance(const std::vector<std::array<double, 4>> &a,
                   const std::vector<std::array<double, 4>> &b);

AssignmentResult LinearAssignment(const Matrix &cost_matrix, double thresh);

Matrix FuseScore(const Matrix &cost_matrix,
                 const std::vector<std::shared_ptr<STrack>> &detections);

}  // namespace fasttracker

