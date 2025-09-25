#include "fasttracker/fasttracker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>

namespace fasttracker {

namespace {

double ComputeIoU(const std::array<double, 4> &a,
                  const std::array<double, 4> &b) {
  double inter_x1 = std::max(a[0], b[0]);
  double inter_y1 = std::max(a[1], b[1]);
  double inter_x2 = std::min(a[2], b[2]);
  double inter_y2 = std::min(a[3], b[3]);
  double iw = std::max(0.0, inter_x2 - inter_x1);
  double ih = std::max(0.0, inter_y2 - inter_y1);
  double inter = iw * ih;
  if (inter <= 0.0) {
    return 0.0;
  }
  double area_a = (a[2] - a[0]) * (a[3] - a[1]);
  double area_b = (b[2] - b[0]) * (b[3] - b[1]);
  double denom = area_a + area_b - inter;
  if (denom <= 0.0) {
    return 0.0;
  }
  return inter / denom;
}

}  // namespace

Fasttracker::Fasttracker(const FasttrackerArgs &args,
                         const FasttrackerConfig &config, double frame_rate)
    : args_(args), det_thresh_(config.track_thresh),
      match_thresh_(config.match_thresh),
      buffer_size_(static_cast<int>(frame_rate / 30.0 * config.track_buffer)),
      max_time_lost_(buffer_size_),
      reset_velocity_offset_occ_(config.reset_velocity_offset_occ),
      reset_pos_offset_occ_(config.reset_pos_offset_occ),
      enlarge_bbox_occ_(config.enlarge_bbox_occ),
      dampen_motion_occ_(config.dampen_motion_occ),
      active_occ_to_lost_thresh_(config.active_occ_to_lost_thresh),
      init_iou_suppress_(config.init_iou_suppress) {
  std::cout << "=== FastTracker Config ===\n";
  std::cout << "  track_thresh: " << det_thresh_ << "\n";
  std::cout << "  match_thresh: " << match_thresh_ << "\n";
  std::cout << "  track_buffer: " << buffer_size_ << "\n";
  std::cout << "  reset_velocity_offset_occ: " << reset_velocity_offset_occ_ << "\n";
  std::cout << "  reset_pos_offset_occ: " << reset_pos_offset_occ_ << "\n";
  std::cout << "  enlarge_bbox_occ: " << enlarge_bbox_occ_ << "\n";
  std::cout << "  dampen_motion_occ: " << dampen_motion_occ_ << "\n";
  std::cout << "  active_occ_to_lost_thresh: " << active_occ_to_lost_thresh_ << "\n";
  std::cout << "  init_iou_suppress: " << init_iou_suppress_ << "\n";
  std::cout << "=============================\n";
}

std::vector<std::shared_ptr<STrack>> Fasttracker::Update(
    const std::vector<Detection> &output_results,
    const std::array<int, 2> &img_info, const std::array<int, 2> &img_size) {
  ++frame_id_;
  std::vector<std::shared_ptr<STrack>> activated_stracks;
  std::vector<std::shared_ptr<STrack>> refind_stracks;
  std::vector<std::shared_ptr<STrack>> lost_stracks;
  std::vector<std::shared_ptr<STrack>> removed_stracks;

  std::vector<std::array<double, 4>> bboxes(output_results.size());
  std::vector<double> scores(output_results.size(), 0.0);
  for (std::size_t i = 0; i < output_results.size(); ++i) {
    bboxes[i] = output_results[i].tlbr;
    scores[i] = output_results[i].CombinedScore();
  }

  double img_h = static_cast<double>(img_info[0]);
  double img_w = static_cast<double>(img_info[1]);
  double scale = 1.0;
  if (img_h > 0.0 && img_w > 0.0) {
    scale = std::min(img_size[0] / img_h, img_size[1] / img_w);
  }
  if (scale > 0.0 && std::abs(scale - 1.0) > 1e-6) {
    for (auto &bbox : bboxes) {
      for (double &value : bbox) {
        value /= scale;
      }
    }
  }

  std::vector<int> remain_inds;
  std::vector<int> inds_low;
  std::vector<int> inds_second;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > det_thresh_) {
      remain_inds.push_back(static_cast<int>(i));
    }
    if (scores[i] > 0.25) {
      inds_low.push_back(static_cast<int>(i));
    }
    if (scores[i] < det_thresh_ && scores[i] > 0.25) {
      inds_second.push_back(static_cast<int>(i));
    }
  }

  std::vector<std::shared_ptr<STrack>> detections;
  for (int idx : remain_inds) {
    auto det = std::make_shared<STrack>(
        STrack::TlbrToTlwh(bboxes[idx]), scores[idx]);
    detections.push_back(det);
  }

  std::vector<std::shared_ptr<STrack>> detections_second;
  for (int idx : inds_second) {
    auto det = std::make_shared<STrack>(
        STrack::TlbrToTlwh(bboxes[idx]), scores[idx]);
    detections_second.push_back(det);
  }

  std::vector<std::shared_ptr<STrack>> unconfirmed;
  std::vector<std::shared_ptr<STrack>> tracked_stracks;
  for (const auto &track : tracked_stracks_) {
    if (!track->is_activated_) {
      unconfirmed.push_back(track);
    } else {
      tracked_stracks.push_back(track);
    }
  }

  auto strack_pool = JointStracks(tracked_stracks, lost_stracks_);
  STrack::MultiPredict(strack_pool);

  Matrix dists = IouDistance(strack_pool, detections);
  if (!args_.mot20) {
    dists = FuseScore(dists, detections);
  }
  AssignmentResult matches = LinearAssignment(dists, match_thresh_);

  for (const auto &pair : matches.matches) {
    auto track = strack_pool[pair.first];
    auto det = detections[pair.second];
    if (track->state_ == TrackState::Tracked) {
      track->Update(*det, frame_id_);
      activated_stracks.push_back(track);
    } else {
      track->ReActivate(*det, frame_id_, false);
      refind_stracks.push_back(track);
    }
    track->is_occluded = false;
    track->not_matched = 0;
    track->occluded_len = 0;
  }

  std::vector<int> unmatch_track = matches.unmatched_rows;
  std::vector<int> unmatch_detection = matches.unmatched_cols;

  std::vector<std::shared_ptr<STrack>> r_tracked_stracks;
  std::vector<int> r_tracked_indices;
  for (int idx : unmatch_track) {
    if (strack_pool[idx]->state_ == TrackState::Tracked) {
      r_tracked_indices.push_back(idx);
      r_tracked_stracks.push_back(strack_pool[idx]);
    }
  }

  Matrix dists_second = IouDistance(r_tracked_stracks, detections_second);
  AssignmentResult matches_second =
      LinearAssignment(dists_second, 0.5);
  for (const auto &pair : matches_second.matches) {
    auto track = r_tracked_stracks[pair.first];
    auto det = detections_second[pair.second];
    if (track->state_ == TrackState::Tracked) {
      track->Update(*det, frame_id_);
      activated_stracks.push_back(track);
    } else {
      track->ReActivate(*det, frame_id_, false);
      refind_stracks.push_back(track);
    }
    track->is_occluded = false;
    track->not_matched = 0;
    track->occluded_len = 0;
  }

  std::vector<int> occlusion_candidates;
  for (int idx : matches_second.unmatched_rows) {
    occlusion_candidates.push_back(r_tracked_indices[idx]);
  }

  for (int idx : occlusion_candidates) {
    auto track = strack_pool[idx];
    track->not_matched += 1;
    if (!track->is_occluded && track->state_ == TrackState::Tracked) {
      for (const auto &other : activated_stracks) {
        if (other->track_id() == track->track_id()) {
          continue;
        }
        if (!other->is_activated_ || other->is_occluded) {
          continue;
        }
        if (IsOccludedBy(track->Tlbr(), other->Tlbr(), 0.7)) {
          track->is_occluded = true;
          track->occluded_len += 1;
          track->last_occluded_frame = frame_id_;
          track->was_recently_occluded = true;

          if (reset_velocity_offset_occ_ > 0 &&
              track->mean_history.size() >=
                  static_cast<std::size_t>(reset_velocity_offset_occ_)) {
            const auto &old_mean = track->mean_history[
                track->mean_history.size() - reset_velocity_offset_occ_];
            for (int i = 4; i < 8; ++i) {
              track->mean_[i] = old_mean[i];
            }
          }
          if (reset_pos_offset_occ_ > 0 &&
              track->mean_history.size() >=
                  static_cast<std::size_t>(reset_pos_offset_occ_)) {
            const auto &old_mean = track->mean_history[
                track->mean_history.size() - reset_pos_offset_occ_];
            for (int i = 0; i < 4; ++i) {
              track->mean_[i] = old_mean[i];
            }
          }
          if (track->occluded_len == 1 && track->mean_.size() >= 4) {
            track->mean_[3] *= enlarge_bbox_occ_;
          }
          for (int i = 4; i < 8 && i < static_cast<int>(track->mean_.size());
               ++i) {
            track->mean_[i] *= dampen_motion_occ_;
          }
          break;
        }
      }
    }

    if (!track->is_occluded) {
      track->occluded_len = 0;
    } else {
      track->occluded_len += 1;
    }

    if (track->was_recently_occluded &&
        frame_id_ - track->last_occluded_frame > 40) {
      track->was_recently_occluded = false;
    }

    if (track->state_ != TrackState::Lost) {
      if (track->not_matched > 2 &&
          (!track->is_occluded ||
           track->occluded_len > active_occ_to_lost_thresh_)) {
        track->MarkLost();
        lost_stracks.push_back(track);
      }
    }
  }

  std::vector<std::shared_ptr<STrack>> detections_unmatched;
  for (int idx : unmatch_detection) {
    if (idx >= 0 && idx < static_cast<int>(detections.size())) {
      detections_unmatched.push_back(detections[idx]);
    }
  }

  Matrix dists_unconfirmed = IouDistance(unconfirmed, detections_unmatched);
  if (!args_.mot20) {
    dists_unconfirmed = FuseScore(dists_unconfirmed, detections_unmatched);
  }
  AssignmentResult matches_unconfirmed = LinearAssignment(dists_unconfirmed, 0.7);
  for (const auto &pair : matches_unconfirmed.matches) {
    auto track = unconfirmed[pair.first];
    track->Update(*detections_unmatched[pair.second], frame_id_);
    activated_stracks.push_back(track);
  }
  for (int idx : matches_unconfirmed.unmatched_rows) {
    auto track = unconfirmed[idx];
    track->MarkLost();
    lost_stracks.push_back(track);
  }

  std::unordered_map<int, std::shared_ptr<STrack>> active_now_map;
  for (const auto &track : tracked_stracks_) {
    if (track->state_ == TrackState::Tracked) {
      active_now_map[track->track_id()] = track;
    }
  }
  for (const auto &track : activated_stracks) {
    active_now_map[track->track_id()] = track;
  }
  std::vector<std::shared_ptr<STrack>> active_now;
  active_now.reserve(active_now_map.size());
  for (const auto &kv : active_now_map) {
    active_now.push_back(kv.second);
  }

  for (int idx : matches_unconfirmed.unmatched_cols) {
    auto track = detections_unmatched[idx];
    if (track->score() < det_thresh_) {
      continue;
    }
    std::array<double, 4> det_box = STrack::TlwhToTlbr(track->Tlwh());
    double max_iou = 0.0;
    for (const auto &active : active_now) {
      max_iou = std::max(max_iou, ComputeIoU(det_box, active->Tlbr()));
      if (max_iou >= init_iou_suppress_) {
        break;
      }
    }
    if (max_iou < init_iou_suppress_) {
      track->Activate(&kalman_filter_, frame_id_);
      activated_stracks.push_back(track);
    }
  }

  for (const auto &track : lost_stracks_) {
    bool recently_occluded =
        track->was_recently_occluded &&
        (frame_id_ - track->last_occluded_frame <= 40);
    if (!recently_occluded &&
        (frame_id_ - track->frame_id()) > max_time_lost_) {
      track->MarkRemoved();
      removed_stracks.push_back(track);
    }
  }

  std::vector<std::shared_ptr<STrack>> tracked_tracked;
  for (const auto &track : tracked_stracks_) {
    if (track->state_ == TrackState::Tracked) {
      tracked_tracked.push_back(track);
    }
  }

  tracked_stracks_ = JointStracks(tracked_tracked, activated_stracks);
  tracked_stracks_ = JointStracks(tracked_stracks_, refind_stracks);

  lost_stracks_ = SubStracks(lost_stracks_, tracked_stracks_);
  lost_stracks_.insert(lost_stracks_.end(), lost_stracks.begin(),
                       lost_stracks.end());
  lost_stracks_ = SubStracks(lost_stracks_, removed_stracks_);
  removed_stracks_.insert(removed_stracks_.end(), removed_stracks.begin(),
                          removed_stracks.end());

  auto filtered = RemoveDuplicateStracks(tracked_stracks_, lost_stracks_);
  tracked_stracks_ = filtered.first;
  lost_stracks_ = filtered.second;

  std::vector<std::shared_ptr<STrack>> output;
  for (const auto &track : tracked_stracks_) {
    if (track->is_activated_) {
      output.push_back(track);
    }
  }
  return output;
}

bool Fasttracker::IsOccludedBy(const std::array<double, 4> &a,
                               const std::array<double, 4> &b,
                               double iou_thresh) {
  double inter_x1 = std::max(a[0], b[0]);
  double inter_y1 = std::max(a[1], b[1]);
  double inter_x2 = std::min(a[2], b[2]);
  double inter_y2 = std::min(a[3], b[3]);
  double iw = std::max(0.0, inter_x2 - inter_x1);
  double ih = std::max(0.0, inter_y2 - inter_y1);
  double inter = iw * ih;
  double area_a = (a[2] - a[0]) * (a[3] - a[1]);
  if (area_a <= 0.0) {
    return false;
  }
  double iou = inter / area_a;
  return iou > iou_thresh;
}

double Fasttracker::IoU(const std::array<double, 4> &a,
                        const std::array<double, 4> &b) {
  return ComputeIoU(a, b);
}

std::vector<std::shared_ptr<STrack>> JointStracks(
    const std::vector<std::shared_ptr<STrack>> &a,
    const std::vector<std::shared_ptr<STrack>> &b) {
  std::unordered_map<int, std::shared_ptr<STrack>> exists;
  std::vector<std::shared_ptr<STrack>> result;
  result.reserve(a.size() + b.size());
  for (const auto &track : a) {
    exists[track->track_id()] = track;
    result.push_back(track);
  }
  for (const auto &track : b) {
    if (exists.find(track->track_id()) == exists.end()) {
      exists[track->track_id()] = track;
      result.push_back(track);
    }
  }
  return result;
}

std::vector<std::shared_ptr<STrack>> SubStracks(
    const std::vector<std::shared_ptr<STrack>> &a,
    const std::vector<std::shared_ptr<STrack>> &b) {
  std::unordered_map<int, std::shared_ptr<STrack>> track_map;
  for (const auto &track : a) {
    track_map[track->track_id()] = track;
  }
  for (const auto &track : b) {
    track_map.erase(track->track_id());
  }
  std::vector<std::shared_ptr<STrack>> result;
  result.reserve(track_map.size());
  for (const auto &kv : track_map) {
    result.push_back(kv.second);
  }
  return result;
}

std::pair<std::vector<std::shared_ptr<STrack>>,
          std::vector<std::shared_ptr<STrack>>>
RemoveDuplicateStracks(const std::vector<std::shared_ptr<STrack>> &a,
                       const std::vector<std::shared_ptr<STrack>> &b) {
  Matrix pdist = IouDistance(a, b);
  std::vector<int> dup_a;
  std::vector<int> dup_b;
  for (std::size_t i = 0; i < pdist.rows(); ++i) {
    for (std::size_t j = 0; j < pdist.cols(); ++j) {
      if (pdist(i, j) < 0.15) {
        int time_a = a[i]->frame_id() - a[i]->start_frame();
        int time_b = b[j]->frame_id() - b[j]->start_frame();
        if (time_a > time_b) {
          dup_b.push_back(static_cast<int>(j));
        } else {
          dup_a.push_back(static_cast<int>(i));
        }
      }
    }
  }

  std::vector<std::shared_ptr<STrack>> res_a;
  std::vector<std::shared_ptr<STrack>> res_b;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::find(dup_a.begin(), dup_a.end(), static_cast<int>(i)) ==
        dup_a.end()) {
      res_a.push_back(a[i]);
    }
  }
  for (std::size_t j = 0; j < b.size(); ++j) {
    if (std::find(dup_b.begin(), dup_b.end(), static_cast<int>(j)) ==
        dup_b.end()) {
      res_b.push_back(b[j]);
    }
  }
  return {res_a, res_b};
}

}  // namespace fasttracker

