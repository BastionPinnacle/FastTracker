#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "fasttracker/matching.hpp"
#include "fasttracker/strack.hpp"

namespace fasttracker {

struct FasttrackerArgs {
  bool mot20 = false;
};

struct FasttrackerConfig {
  double track_thresh = 0.5;
  double match_thresh = 0.8;
  double track_buffer = 30.0;
  int reset_velocity_offset_occ = 5;
  int reset_pos_offset_occ = 10;
  double enlarge_bbox_occ = 1.05;
  double dampen_motion_occ = 0.5;
  int active_occ_to_lost_thresh = 10;
  double init_iou_suppress = 0.8;
};

struct Detection {
  std::array<double, 4> tlbr;
  double object_score = 1.0;
  double class_score = 1.0;

  double CombinedScore() const { return object_score * class_score; }
};

class Fasttracker {
public:
  Fasttracker(const FasttrackerArgs &args, const FasttrackerConfig &config,
              double frame_rate = 30.0);

  std::vector<std::shared_ptr<STrack>> Update(
      const std::vector<Detection> &output_results,
      const std::array<int, 2> &img_info, const std::array<int, 2> &img_size);

  const std::vector<std::shared_ptr<STrack>> &tracked() const {
    return tracked_stracks_;
  }

private:
  static bool IsOccludedBy(const std::array<double, 4> &a,
                           const std::array<double, 4> &b, double iou_thresh);
  static double IoU(const std::array<double, 4> &a,
                    const std::array<double, 4> &b);

  std::vector<std::shared_ptr<STrack>> tracked_stracks_;
  std::vector<std::shared_ptr<STrack>> lost_stracks_;
  std::vector<std::shared_ptr<STrack>> removed_stracks_;

  int frame_id_ = 0;
  FasttrackerArgs args_;

  double det_thresh_;
  double match_thresh_;
  int buffer_size_;
  int max_time_lost_;

  int reset_velocity_offset_occ_;
  int reset_pos_offset_occ_;
  double enlarge_bbox_occ_;
  double dampen_motion_occ_;
  int active_occ_to_lost_thresh_;
  double init_iou_suppress_;

  KalmanFilter kalman_filter_;
};

std::vector<std::shared_ptr<STrack>> JointStracks(
    const std::vector<std::shared_ptr<STrack>> &a,
    const std::vector<std::shared_ptr<STrack>> &b);

std::vector<std::shared_ptr<STrack>> SubStracks(
    const std::vector<std::shared_ptr<STrack>> &a,
    const std::vector<std::shared_ptr<STrack>> &b);

std::pair<std::vector<std::shared_ptr<STrack>>,
          std::vector<std::shared_ptr<STrack>>>
RemoveDuplicateStracks(const std::vector<std::shared_ptr<STrack>> &a,
                       const std::vector<std::shared_ptr<STrack>> &b);

}  // namespace fasttracker

