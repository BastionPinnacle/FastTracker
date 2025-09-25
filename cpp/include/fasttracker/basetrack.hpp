#pragma once

#include <memory>

namespace fasttracker {

enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

class Fasttracker;

class BaseTrack {
 public:
  BaseTrack();
  virtual ~BaseTrack() = default;

  int track_id() const { return track_id_; }
  bool is_activated() const { return is_activated_; }
  TrackState state() const { return state_; }
  int frame_id() const { return frame_id_; }
  int start_frame() const { return start_frame_; }

  void MarkLost();
  void MarkRemoved();

 protected:
  static int NextId();

  int track_id_;
  bool is_activated_;
  TrackState state_;
  int start_frame_;
  int frame_id_;
  int time_since_update_;

  friend class Fasttracker;
};

using BaseTrackPtr = std::shared_ptr<BaseTrack>;

}  // namespace fasttracker

