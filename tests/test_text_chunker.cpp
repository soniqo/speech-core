#include "speech_core/util/text_chunker.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace speech_core;

// Deterministic stand-in for the phonemizer: one token per non-space
// character plus two for BOS/EOS, mirroring how real tokenizers charge a
// fixed per-call overhead.
static size_t fake_count(const std::string& text) {
    size_t n = 2;
    for (char c : text) {
        if (c != ' ' && c != '\t') n++;
    }
    return n;
}

static std::vector<std::string> chunk(const std::string& text,
                                      size_t budget,
                                      size_t hard_cap = 128,
                                      size_t min_tail = 12) {
    return chunk_text_for_synthesis(text, fake_count, budget, hard_cap,
                                    min_tail);
}

void test_empty_and_whitespace() {
    assert(chunk("", 72).empty());
    assert(chunk("   \n\t  ", 72).empty());
    printf("  PASS: empty_and_whitespace\n");
}

void test_short_text_single_chunk() {
    auto chunks = chunk("Music stopped.", 72);
    assert(chunks.size() == 1);
    assert(chunks[0] == "Music stopped.");
    printf("  PASS: short_text_single_chunk\n");
}

void test_sentences_pack_together() {
    // Two short sentences fit one budget: they must not be split apart.
    auto chunks = chunk("Calling Anna. Please wait.", 72);
    assert(chunks.size() == 1);
    assert(chunks[0] == "Calling Anna. Please wait.");
    printf("  PASS: sentences_pack_together\n");
}

void test_splits_on_sentence_boundary() {
    std::string a = "The first sentence is fairly long already.";
    std::string b = "The second sentence is also fairly long.";
    size_t budget = fake_count(a) + 4;  // one fits, both do not
    auto chunks = chunk(a + " " + b, budget);
    assert(chunks.size() == 2);
    assert(chunks[0] == a);
    assert(chunks[1] == b);
    printf("  PASS: splits_on_sentence_boundary\n");
}

void test_budget_respected() {
    std::string text =
        "I can call your contacts, dial numbers, look up phone numbers, "
        "play or stop music, and set the volume, all offline, on this "
        "device. Ask me anything you like. There is more where that came "
        "from!";
    size_t budget = 72;
    auto chunks = chunk(text, budget);
    assert(chunks.size() >= 2);
    for (const auto& c : chunks) {
        assert(fake_count(c) <= budget);
        assert(!c.empty());
    }
    printf("  PASS: budget_respected\n");
}

void test_clause_fallback_for_run_on_sentence() {
    // No sentence enders: must fall back to clause boundaries.
    std::string text =
        "first clause goes here, second clause goes here, third clause "
        "goes here, fourth clause goes here";
    size_t budget = 30;
    auto chunks = chunk(text, budget);
    assert(chunks.size() >= 2);
    for (const auto& c : chunks) assert(fake_count(c) <= budget);
    printf("  PASS: clause_fallback_for_run_on_sentence\n");
}

void test_word_fallback() {
    // No punctuation at all.
    std::string text = "one two three four five six seven eight nine ten";
    size_t budget = 12;
    auto chunks = chunk(text, budget, 128, 3);
    assert(chunks.size() >= 3);
    for (const auto& c : chunks) assert(fake_count(c) <= budget);
    printf("  PASS: word_fallback\n");
}

void test_monster_word_char_split() {
    std::string word(50, 'x');
    size_t budget = 12;
    auto chunks = chunk(word, budget, 128, 3);
    assert(chunks.size() >= 4);
    std::string rejoined;
    for (const auto& c : chunks) {
        assert(fake_count(c) <= budget);
        rejoined += c;
    }
    assert(rejoined == word);
    printf("  PASS: monster_word_char_split\n");
}

void test_utf8_never_split_mid_character() {
    // Two-byte characters (ü, é) and a three-byte character (中) repeated
    // into an unbroken word; chunks must never begin with a continuation
    // byte (0b10xxxxxx).
    std::string word;
    for (int i = 0; i < 12; i++) word += "\xC3\xBC\xE4\xB8\xAD";  // ü中
    auto chunks = chunk(word, 10, 128, 2);
    assert(chunks.size() >= 2);
    std::string rejoined;
    for (const auto& c : chunks) {
        assert(!c.empty());
        assert((static_cast<unsigned char>(c[0]) & 0xC0) != 0x80);
        rejoined += c;
    }
    assert(rejoined == word);
    printf("  PASS: utf8_never_split_mid_character\n");
}

void test_tiny_tail_merges() {
    std::string a = "This is a reasonably sized first sentence here.";
    std::string b = "Ok.";
    size_t budget = fake_count(a) + 2;  // tail cannot pack into the chunk
    auto chunks = chunk(a + " " + b, budget, /*hard_cap=*/128,
                        /*min_tail=*/12);
    // "Ok." is below min_tail and the hard cap allows the merge.
    assert(chunks.size() == 1);
    assert(chunks[0] == a + " " + b);
    printf("  PASS: tiny_tail_merges\n");
}

void test_tiny_tail_kept_when_hard_cap_blocks() {
    std::string a = "This is a reasonably sized first sentence here.";
    std::string b = "Ok.";
    size_t budget = fake_count(a) + 2;
    // Hard cap equal to the budget: the merge would overflow, keep the tail.
    auto chunks = chunk(a + " " + b, budget, /*hard_cap=*/budget,
                        /*min_tail=*/12);
    assert(chunks.size() == 2);
    assert(chunks[1] == b);
    printf("  PASS: tiny_tail_kept_when_hard_cap_blocks\n");
}

void test_newlines_are_boundaries() {
    auto chunks = chunk("Line one has words\nLine two has words", 18);
    assert(chunks.size() == 2);
    printf("  PASS: newlines_are_boundaries\n");
}

int main() {
    printf("test_text_chunker:\n");
    test_empty_and_whitespace();
    test_short_text_single_chunk();
    test_sentences_pack_together();
    test_splits_on_sentence_boundary();
    test_budget_respected();
    test_clause_fallback_for_run_on_sentence();
    test_word_fallback();
    test_monster_word_char_split();
    test_utf8_never_split_mid_character();
    test_tiny_tail_merges();
    test_newlines_are_boundaries();
    test_tiny_tail_kept_when_hard_cap_blocks();
    printf("All text_chunker tests passed.\n");
    return 0;
}
