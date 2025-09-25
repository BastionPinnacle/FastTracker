#include "fasttracker/matching.hpp"

#include <algorithm>
#include <limits>

namespace fasttracker {

namespace {

double IoU(const std::array<double, 4> &a, const std::array<double, 4> &b) {
  double ax1 = a[0];
  double ay1 = a[1];
  double ax2 = a[2];
  double ay2 = a[3];
  double bx1 = b[0];
  double by1 = b[1];
  double bx2 = b[2];
  double by2 = b[3];

  double inter_x1 = std::max(ax1, bx1);
  double inter_y1 = std::max(ay1, by1);
  double inter_x2 = std::min(ax2, bx2);
  double inter_y2 = std::min(ay2, by2);
  double iw = std::max(0.0, inter_x2 - inter_x1);
  double ih = std::max(0.0, inter_y2 - inter_y1);
  double inter = iw * ih;
  if (inter <= 0.0) {
    return 0.0;
  }
  double area_a = (ax2 - ax1) * (ay2 - ay1);
  double area_b = (bx2 - bx1) * (by2 - by1);
  double denom = area_a + area_b - inter;
  if (denom <= 0.0) {
    return 0.0;
  }
  return inter / denom;
}

Matrix BuildCost(const std::vector<std::array<double, 4>> &a,
                 const std::vector<std::array<double, 4>> &b) {
  Matrix cost(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t j = 0; j < b.size(); ++j) {
      cost(i, j) = 1.0 - IoU(a[i], b[j]);
    }
  }
  return cost;
}

}  // namespace

Matrix IouDistance(const std::vector<std::shared_ptr<STrack>> &a,
                   const std::vector<std::shared_ptr<STrack>> &b) {
  std::vector<std::array<double, 4>> atlbrs;
  std::vector<std::array<double, 4>> btlbrs;
  atlbrs.reserve(a.size());
  btlbrs.reserve(b.size());
  for (const auto &track : a) {
    atlbrs.push_back(track->Tlbr());
  }
  for (const auto &track : b) {
    btlbrs.push_back(track->Tlbr());
  }
  return BuildCost(atlbrs, btlbrs);
}

Matrix IouDistance(const std::vector<std::array<double, 4>> &a,
                   const std::vector<std::array<double, 4>> &b) {
  return BuildCost(a, b);
}

AssignmentResult LinearAssignment(const Matrix &cost_matrix, double thresh) {
  AssignmentResult result;
  int rows = static_cast<int>(cost_matrix.rows());
  int cols = static_cast<int>(cost_matrix.cols());
  if (rows == 0 || cols == 0) {
    for (int i = 0; i < rows; ++i) {
      result.unmatched_rows.push_back(i);
    }
    for (int j = 0; j < cols; ++j) {
      result.unmatched_cols.push_back(j);
    }
    return result;
  }

  int n = std::max(rows, cols);
  std::vector<std::vector<double>> cost(n, std::vector<double>(n, thresh + 1.0));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      cost[i][j] = cost_matrix(i, j);
    }
  }

  std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
    std::vector<bool> used(n + 1, false);
    do {
      used[j0] = true;
      int i0 = p[j0];
      double delta = std::numeric_limits<double>::infinity();
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[j]) {
          continue;
        }
        double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> assignment(rows, -1);
  for (int j = 1; j <= n; ++j) {
    if (p[j] <= rows && j <= cols) {
      assignment[p[j] - 1] = j - 1;
    }
  }

  for (int i = 0; i < rows; ++i) {
    int j = assignment[i];
    if (j >= 0 && cost_matrix(i, j) <= thresh) {
      result.matches.emplace_back(i, j);
    } else {
      result.unmatched_rows.push_back(i);
    }
  }
  std::vector<bool> matched_cols(cols, false);
  for (const auto &match : result.matches) {
    matched_cols[match.second] = true;
  }
  for (int j = 0; j < cols; ++j) {
    if (!matched_cols[j]) {
      result.unmatched_cols.push_back(j);
    }
  }

  return result;
}

Matrix FuseScore(const Matrix &cost_matrix,
                 const std::vector<std::shared_ptr<STrack>> &detections) {
  Matrix result(cost_matrix.rows(), cost_matrix.cols());
  for (std::size_t i = 0; i < cost_matrix.rows(); ++i) {
    for (std::size_t j = 0; j < cost_matrix.cols(); ++j) {
      double iou_sim = 1.0 - cost_matrix(i, j);
      double det_score = detections[j]->score();
      double fuse_sim = iou_sim * det_score;
      result(i, j) = 1.0 - fuse_sim;
    }
  }
  return result;
}

}  // namespace fasttracker

