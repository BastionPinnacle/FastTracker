#pragma once

#include <array>
#include <deque>
#include <memory>
#include <vector>

#include "fasttracker/basetrack.hpp"
#include "fasttracker/kalman_filter.hpp"

namespace fasttracker {

class Fasttracker;

class STrack : public BaseTrack {
public:
  explicit STrack(const std::array<double, 4> &tlwh, double score);

  void Predict();
  void Activate(KalmanFilter *filter, int frame_id);
  void ReActivate(const STrack &new_track, int frame_id, bool new_id = false);
  void Update(const STrack &new_track, int frame_id);

  const std::vector<double> &mean() const { return mean_; }
  const Matrix &covariance() const { return covariance_; }
  void set_mean(const std::vector<double> &mean) { mean_ = mean; }
  void set_covariance(const Matrix &covariance) { covariance_ = covariance; }

  double score() const { return score_; }
  void set_score(double score) { score_ = score; }

  std::array<double, 4> Tlwh() const;
  std::array<double, 4> Tlbr() const;

  static std::array<double, 4> TlbrToTlwh(const std::array<double, 4> &tlbr);
  static std::array<double, 4> TlwhToTlbr(const std::array<double, 4> &tlwh);
  static std::vector<double> TlwhToXyah(const std::array<double, 4> &tlwh);

  static void MultiPredict(const std::vector<std::shared_ptr<STrack>> &tracks);

  int not_matched = 0;
  bool is_occluded = false;
  int occluded_len = 0;
  int last_occluded_frame = -1;
  bool was_recently_occluded = false;

  std::vector<std::vector<double>> mean_history;

private:
  std::array<double, 4> tlwh_;
  KalmanFilter *kalman_filter_;
  std::vector<double> mean_;
  Matrix covariance_;
  double score_;
  int tracklet_len_;
  bool mean_valid_ = false;

  friend class Fasttracker;
};

using STrackPtr = std::shared_ptr<STrack>;

}  // namespace fasttracker

