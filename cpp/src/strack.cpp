#include "fasttracker/strack.hpp"

#include <algorithm>

namespace fasttracker {

namespace {
constexpr int kMaxHistory = 100;
}

STrack::STrack(const std::array<double, 4> &tlwh, double score)
    : tlwh_(tlwh), kalman_filter_(nullptr), mean_(), covariance_(8, 8),
      score_(score), tracklet_len_(0), mean_valid_(false) {}

void STrack::Predict() {
  if (!kalman_filter_ || !mean_valid_) {
    return;
  }
  std::vector<double> mean_copy = mean_;
  if (state_ != TrackState::Tracked && mean_copy.size() >= 8) {
    mean_copy[7] = 0.0;
  }
  auto predicted = kalman_filter_->Predict(mean_copy, covariance_);
  mean_ = predicted.first;
  covariance_ = predicted.second;
}

void STrack::Activate(KalmanFilter *filter, int frame_id) {
  kalman_filter_ = filter;
  track_id_ = NextId();
  auto initiated = kalman_filter_->Initiate(TlwhToXyah(tlwh_));
  mean_ = initiated.first;
  covariance_ = initiated.second;
  mean_valid_ = true;

  mean_history.push_back(mean_);
  if (mean_history.size() > kMaxHistory) {
    mean_history.erase(mean_history.begin());
  }

  tracklet_len_ = 0;
  state_ = TrackState::Tracked;
  is_activated_ = (frame_id == 1);
  frame_id_ = frame_id;
  start_frame_ = frame_id;
}

void STrack::ReActivate(const STrack &new_track, int frame_id, bool new_id) {
  if (!kalman_filter_) {
    kalman_filter_ = new_track.kalman_filter_;
  }
  auto updated = kalman_filter_->Update(
      mean_, covariance_, TlwhToXyah(new_track.Tlwh()));
  mean_ = updated.first;
  covariance_ = updated.second;
  mean_valid_ = true;

  mean_history.push_back(mean_);
  if (mean_history.size() > kMaxHistory) {
    mean_history.erase(mean_history.begin());
  }

  tracklet_len_ = 0;
  state_ = TrackState::Tracked;
  is_activated_ = true;
  frame_id_ = frame_id;
  if (new_id) {
    track_id_ = NextId();
  }
  score_ = new_track.score_;
}

void STrack::Update(const STrack &new_track, int frame_id) {
  if (!kalman_filter_) {
    kalman_filter_ = new_track.kalman_filter_;
  }
  frame_id_ = frame_id;
  ++tracklet_len_;
  auto updated = kalman_filter_->Update(
      mean_, covariance_, TlwhToXyah(new_track.Tlwh()));
  mean_ = updated.first;
  covariance_ = updated.second;
  mean_valid_ = true;

  mean_history.push_back(mean_);
  if (mean_history.size() > kMaxHistory) {
    mean_history.erase(mean_history.begin());
  }

  state_ = TrackState::Tracked;
  is_activated_ = true;
  score_ = new_track.score_;
}

std::array<double, 4> STrack::Tlwh() const {
  if (!mean_valid_) {
    return tlwh_;
  }
  std::array<double, 4> ret = {mean_[0], mean_[1], mean_[2], mean_[3]};
  ret[2] *= ret[3];
  ret[0] -= ret[2] / 2.0;
  ret[1] -= ret[3] / 2.0;
  return ret;
}

std::array<double, 4> STrack::Tlbr() const {
  std::array<double, 4> ret = Tlwh();
  ret[2] += ret[0];
  ret[3] += ret[1];
  return ret;
}

std::array<double, 4> STrack::TlbrToTlwh(const std::array<double, 4> &tlbr) {
  return {tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]};
}

std::array<double, 4> STrack::TlwhToTlbr(const std::array<double, 4> &tlwh) {
  return {tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]};
}

std::vector<double> STrack::TlwhToXyah(const std::array<double, 4> &tlwh) {
  std::vector<double> ret(tlwh.begin(), tlwh.end());
  ret[0] += ret[2] / 2.0;
  ret[1] += ret[3] / 2.0;
  ret[2] = (ret[3] == 0.0) ? 0.0 : ret[2] / ret[3];
  return ret;
}

void STrack::MultiPredict(const std::vector<std::shared_ptr<STrack>> &tracks) {
  std::vector<std::vector<double>> means;
  std::vector<Matrix> covariances;
  std::vector<std::size_t> indices;
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    if (tracks[i]->mean_valid_) {
      means.push_back(tracks[i]->mean_);
      covariances.push_back(tracks[i]->covariance_);
      indices.push_back(i);
    }
  }
  if (means.empty()) {
    return;
  }
  KalmanFilter filter;
  auto predicted = filter.MultiPredict(means, covariances);
  for (std::size_t idx = 0; idx < indices.size(); ++idx) {
    std::size_t track_index = indices[idx];
    tracks[track_index]->mean_ = predicted.first[idx];
    tracks[track_index]->covariance_ = predicted.second[idx];
  }
}

}  // namespace fasttracker

