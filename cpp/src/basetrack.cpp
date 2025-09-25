#include "fasttracker/basetrack.hpp"

namespace fasttracker {

namespace {
int g_track_counter = 0;
}

BaseTrack::BaseTrack()
    : track_id_(0), is_activated_(false), state_(TrackState::New),
      start_frame_(0), frame_id_(0), time_since_update_(0) {}

int BaseTrack::NextId() { return ++g_track_counter; }

void BaseTrack::MarkLost() { state_ = TrackState::Lost; }

void BaseTrack::MarkRemoved() { state_ = TrackState::Removed; }

}  // namespace fasttracker

