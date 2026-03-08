#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace speech_core {

/// A queued speech output with audio chunks.
struct SpeechItem {
    uint64_t id;
    std::string text;  // source text (for logging/debug)

    enum class State {
        Pending,   ///< Waiting to be played
        Playing,   ///< Currently streaming to speaker
        Paused,    ///< Interrupted, may resume
        Cancelled, ///< Cancelled, will not play
        Done       ///< Fully played
    };

    State state = State::Pending;
};

/// Priority queue for TTS outputs with cancel, interrupt, and resume.
///
/// Manages the ordering and lifecycle of speech outputs. When the user
/// interrupts (barge-in), the current item is paused or cancelled, and
/// the pipeline can resume or discard it based on false-interruption
/// recovery logic.
class SpeechQueue {
public:
    /// Callback when a speech item's state changes.
    using StateCallback = std::function<void(uint64_t id, SpeechItem::State state)>;

    explicit SpeechQueue(StateCallback on_state_change = nullptr);

    /// Enqueue a new speech item. Returns the assigned ID.
    uint64_t enqueue(const std::string& text);

    /// Get the next pending item to play. Returns nullptr if queue is empty.
    SpeechItem* next();

    /// Mark the currently playing item as done.
    void mark_done(uint64_t id);

    /// Pause the currently playing item (for interruption recovery).
    void pause(uint64_t id);

    /// Resume a paused item.
    void resume(uint64_t id);

    /// Cancel a specific item.
    void cancel(uint64_t id);

    /// Cancel all pending and playing items (hard interrupt).
    void cancel_all();

    /// Number of items in the queue (all states).
    size_t size() const;

    /// Whether any item is currently playing.
    bool is_playing() const;

private:
    mutable std::mutex mutex_;
    std::vector<SpeechItem> items_;
    uint64_t next_id_ = 1;
    StateCallback on_state_change_;
};

}  // namespace speech_core
