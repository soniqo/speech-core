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

void test_long_sentence_preserves_whole_words() {
    // A normal sentence with no clause punctuation must fall back to words,
    // not character fragments. This specifically protects hyphenated words
    // in short-turn TTS prompts.
    std::string text =
        "The new short-turn graph is ready for quick replies.";
    auto chunks = chunk(text, /*budget=*/14, /*hard_cap=*/14,
                        /*min_tail=*/0);
    assert(chunks.size() >= 2);

    std::string rejoined;
    bool found_hyphenated_word = false;
    for (const auto& c : chunks) {
        assert(fake_count(c) <= 14);
        if (!rejoined.empty()) rejoined += " ";
        rejoined += c;
        if (c == "short-turn") found_hyphenated_word = true;
    }
    assert(rejoined == text);
    assert(found_hyphenated_word);
    printf("  PASS: long_sentence_preserves_whole_words\n");
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

void test_retry_split_progress_for_13_to_42_tokens() {
    for (size_t parent_tokens = 13; parent_tokens <= 42; ++parent_tokens) {
        std::string word(parent_tokens - 2, 'x');
        auto pieces = split_text_for_synthesis_retry(
            word, fake_count, /*min_tokens=*/6,
            /*max_tokens=*/parent_tokens - 1);
        assert(pieces.size() == 2);
        std::string rejoined;
        for (const auto& piece : pieces) {
            const size_t child_tokens = fake_count(piece);
            assert(!piece.empty());
            assert(child_tokens >= 6);
            assert(child_tokens < parent_tokens);
            rejoined += piece;
        }
        assert(rejoined == word);
    }
    printf("  PASS: retry_split_progress_for_13_to_42_tokens\n");
}

void test_retry_split_utf8_is_balanced_and_intact() {
    std::string word;
    for (int i = 0; i < 14; ++i) word += "\xC3\xBC\xE4\xB8\xAD";  // ü中
    const size_t parent_tokens = fake_count(word);
    auto pieces = split_text_for_synthesis_retry(
        word, fake_count, /*min_tokens=*/6,
        /*max_tokens=*/parent_tokens - 1);
    assert(pieces.size() == 2);
    std::string rejoined;
    for (const auto& piece : pieces) {
        assert(!piece.empty());
        assert((static_cast<unsigned char>(piece[0]) & 0xC0) != 0x80);
        assert(fake_count(piece) >= 6);
        assert(fake_count(piece) < parent_tokens);
        rejoined += piece;
    }
    assert(rejoined == word);
    printf("  PASS: retry_split_utf8_is_balanced_and_intact\n");
}

void test_retry_split_prefers_word_boundary_when_balanced() {
    auto pieces = split_text_for_synthesis_retry(
        "alpha bravo", fake_count, /*min_tokens=*/6, /*max_tokens=*/12);
    assert(pieces.size() == 2);
    assert(pieces[0] == "alpha");
    assert(pieces[1] == "bravo");
    printf("  PASS: retry_split_prefers_word_boundary_when_balanced\n");
}

void test_retry_split_does_not_trade_word_boundary_for_balance() {
    auto pieces = split_text_for_synthesis_retry(
        "alpha bravo charlie", fake_count, /*min_tokens=*/6,
        /*max_tokens=*/16);
    assert(pieces.size() == 2);
    assert(pieces[0] == "alpha bravo");
    assert(pieces[1] == "charlie");
    printf("  PASS: retry_split_does_not_trade_word_boundary_for_balance\n");
}

void test_retry_split_keeps_punctuation_on_left() {
    auto pieces = split_text_for_synthesis_retry(
        "aaaaa. bbbbb", fake_count, /*min_tokens=*/6, /*max_tokens=*/12);
    assert(pieces.size() == 2);
    assert(pieces[0] == "aaaaa.");
    assert(pieces[1] == "bbbbb");
    printf("  PASS: retry_split_keeps_punctuation_on_left\n");
}

void test_retry_split_keeps_closing_quote_with_sentence() {
    auto pieces = split_text_for_synthesis_retry(
        "aaaaa.\" bbbbb", fake_count, /*min_tokens=*/6, /*max_tokens=*/13);
    assert(pieces.size() == 2);
    assert(pieces[0] == "aaaaa.\"");
    assert(pieces[1] == "bbbbb");
    printf("  PASS: retry_split_keeps_closing_quote_with_sentence\n");
}

void test_retry_split_keeps_full_sentence_ender_run() {
    auto pieces = split_text_for_synthesis_retry(
        "aaaaa...?!\" bbbbb", fake_count, /*min_tokens=*/6,
        /*max_tokens=*/18);
    assert(pieces.size() == 2);
    assert(pieces[0] == "aaaaa...?!\"");
    assert(pieces[1] == "bbbbb");
    printf("  PASS: retry_split_keeps_full_sentence_ender_run\n");
}

void test_retry_split_impossible_returns_empty() {
    auto pieces = split_text_for_synthesis_retry(
        "tiny", fake_count, /*min_tokens=*/6, /*max_tokens=*/6);
    assert(pieces.empty());
    printf("  PASS: retry_split_impossible_returns_empty\n");
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
    test_long_sentence_preserves_whole_words();
    test_monster_word_char_split();
    test_utf8_never_split_mid_character();
    test_tiny_tail_merges();
    test_newlines_are_boundaries();
    test_tiny_tail_kept_when_hard_cap_blocks();
    test_retry_split_progress_for_13_to_42_tokens();
    test_retry_split_utf8_is_balanced_and_intact();
    test_retry_split_prefers_word_boundary_when_balanced();
    test_retry_split_does_not_trade_word_boundary_for_balance();
    test_retry_split_keeps_punctuation_on_left();
    test_retry_split_keeps_closing_quote_with_sentence();
    test_retry_split_keeps_full_sentence_ender_run();
    test_retry_split_impossible_returns_empty();
    printf("All text_chunker tests passed.\n");
    return 0;
}
