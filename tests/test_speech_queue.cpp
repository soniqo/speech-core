#include "speech_core/pipeline/speech_queue.h"

#include <cassert>
#include <cstdio>

using namespace speech_core;

void test_enqueue_and_next() {
    SpeechQueue queue;
    auto id1 = queue.enqueue("Hello");
    auto id2 = queue.enqueue("World");

    assert(queue.size() == 2);
    assert(!queue.is_playing());

    auto* item = queue.next();
    assert(item != nullptr);
    assert(item->id == id1);
    assert(item->text == "Hello");
    assert(item->state == SpeechItem::State::Playing);
    assert(queue.is_playing());

    printf("  PASS: enqueue_and_next\n");
}

void test_mark_done() {
    SpeechQueue queue;
    auto id1 = queue.enqueue("Hello");
    queue.enqueue("World");

    queue.next();  // playing id1
    queue.mark_done(id1);

    assert(!queue.is_playing());

    auto* item = queue.next();
    assert(item != nullptr);
    assert(item->text == "World");

    printf("  PASS: mark_done\n");
}

void test_cancel() {
    SpeechQueue queue;
    auto id1 = queue.enqueue("Hello");
    queue.enqueue("World");

    queue.next();  // playing id1
    queue.cancel(id1);

    assert(!queue.is_playing());
    printf("  PASS: cancel\n");
}

void test_cancel_all() {
    SpeechQueue queue;
    queue.enqueue("A");
    queue.enqueue("B");
    queue.enqueue("C");
    queue.next();

    queue.cancel_all();
    assert(!queue.is_playing());
    assert(queue.next() == nullptr);
    printf("  PASS: cancel_all\n");
}

void test_pause_resume() {
    SpeechQueue queue;
    auto id = queue.enqueue("Hello");
    queue.next();
    assert(queue.is_playing());

    queue.pause(id);
    assert(!queue.is_playing());

    queue.resume(id);
    assert(queue.is_playing());

    printf("  PASS: pause_resume\n");
}

void test_state_callback() {
    int callback_count = 0;
    SpeechQueue queue([&callback_count](uint64_t, SpeechItem::State) {
        callback_count++;
    });

    auto id = queue.enqueue("Hello");
    queue.next();       // Pending -> Playing (1)
    queue.pause(id);    // Playing -> Paused (2)
    queue.resume(id);   // Paused -> Playing (3)
    queue.mark_done(id); // Playing -> Done (4)

    assert(callback_count == 4);
    printf("  PASS: state_callback\n");
}

int main() {
    printf("test_speech_queue:\n");
    test_enqueue_and_next();
    test_mark_done();
    test_cancel();
    test_cancel_all();
    test_pause_resume();
    test_state_callback();
    printf("All speech queue tests passed.\n");
    return 0;
}
