#include "speech_core/pipeline/speech_queue.h"

namespace speech_core {

SpeechQueue::SpeechQueue(StateCallback on_state_change)
    : on_state_change_(std::move(on_state_change)) {}

uint64_t SpeechQueue::enqueue(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t id = next_id_++;
    items_.push_back({id, text, SpeechItem::State::Pending});
    return id;
}

SpeechItem* SpeechQueue::next() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : items_) {
        if (item.state == SpeechItem::State::Pending) {
            item.state = SpeechItem::State::Playing;
            if (on_state_change_) on_state_change_(item.id, item.state);
            return &item;
        }
    }
    return nullptr;
}

void SpeechQueue::mark_done(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : items_) {
        if (item.id == id) {
            item.state = SpeechItem::State::Done;
            if (on_state_change_) on_state_change_(id, item.state);
            break;
        }
    }
    // Clean up terminal items from the front. Drain Cancelled too — without
    // this, a Cancelled head blocks all Done cleanup behind it, leaving
    // ghost items in the queue indefinitely (cancel_all + mark_done(later)
    // pattern leaked items_).
    while (!items_.empty() &&
           (items_.front().state == SpeechItem::State::Done ||
            items_.front().state == SpeechItem::State::Cancelled)) {
        items_.erase(items_.begin());
    }
}

void SpeechQueue::pause(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : items_) {
        if (item.id == id && item.state == SpeechItem::State::Playing) {
            item.state = SpeechItem::State::Paused;
            if (on_state_change_) on_state_change_(id, item.state);
            break;
        }
    }
}

void SpeechQueue::resume(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : items_) {
        if (item.id == id && item.state == SpeechItem::State::Paused) {
            item.state = SpeechItem::State::Playing;
            if (on_state_change_) on_state_change_(id, item.state);
            break;
        }
    }
}

void SpeechQueue::cancel(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : items_) {
        if (item.id == id) {
            item.state = SpeechItem::State::Cancelled;
            if (on_state_change_) on_state_change_(id, item.state);
            break;
        }
    }
}

void SpeechQueue::cancel_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : items_) {
        if (item.state == SpeechItem::State::Pending ||
            item.state == SpeechItem::State::Playing ||
            item.state == SpeechItem::State::Paused) {
            item.state = SpeechItem::State::Cancelled;
            if (on_state_change_) on_state_change_(item.id, item.state);
        }
    }
    // Drain — these items are terminal. Without this, the TTS callback
    // for an interrupted item never reaches is_final (the speak() lambda
    // early-returns once state != Speaking), so mark_done is never called
    // and Cancelled entries accumulate in items_ indefinitely.
    items_.clear();
}

size_t SpeechQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return items_.size();
}

bool SpeechQueue::is_playing() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : items_) {
        if (item.state == SpeechItem::State::Playing) return true;
    }
    return false;
}

}  // namespace speech_core
