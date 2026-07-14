#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/models/pocket_tts_tokenizer.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Fixture {
    std::filesystem::path directory;
    std::filesystem::path vocabulary;
    std::filesystem::path scores;

    Fixture() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        directory = std::filesystem::temp_directory_path() /
            ("speech-core-pocket-tokenizer-" + std::to_string(suffix));
        std::filesystem::create_directories(directory);
        vocabulary = directory / "vocab.json";
        scores = directory / "token_scores.json";

        std::ofstream(vocabulary) << R"({
  "<unk>": 0,
  "\u2581hello": 1,
  "\u2581world": 2,
  ".": 3,
  "\u2581": 4,
  "a": 5,
  "\u2581a": 6,
  "<0xC3>": 7,
  "<0xA9>": 8
})";
        std::ofstream(scores) << R"({
  "<unk>": -100.0,
  "\u2581hello": -1.0,
  "\u2581world": -1.0,
  ".": -0.1,
  "\u2581": -0.1,
  "a": -0.1,
  "\u2581a": -1.0,
  "<0xC3>": -2.0,
  "<0xA9>": -2.0
})";
    }

    ~Fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }
};

void test_words_and_whitespace() {
    Fixture fixture;
    speech_core::PocketTtsTokenizer tokenizer(
        fixture.vocabulary.string(), fixture.scores.string());
    assert((tokenizer.encode_ids("hello world.") ==
            std::vector<std::int32_t>{1, 2, 3}));
    assert((tokenizer.encode_tokens("hello world.") ==
            std::vector<std::string>{"\xE2\x96\x81hello", "\xE2\x96\x81world", "."}));
    std::printf("  PASS: words_and_whitespace\n");
}

void test_unigram_scores_choose_best_path() {
    Fixture fixture;
    speech_core::PocketTtsTokenizer tokenizer(
        fixture.vocabulary.string(), fixture.scores.string());
    // Split marker + character scores -0.2, which beats the -1.0 whole token.
    assert((tokenizer.encode_ids("a") == std::vector<std::int32_t>{4, 5}));
    std::printf("  PASS: unigram_scores_choose_best_path\n");
}

void test_utf8_byte_fallback() {
    Fixture fixture;
    speech_core::PocketTtsTokenizer tokenizer(
        fixture.vocabulary.string(), fixture.scores.string());
    assert((tokenizer.encode_ids("\xC3\xA9") ==
            std::vector<std::int32_t>{4, 7, 8}));
    std::printf("  PASS: utf8_byte_fallback\n");
}

void test_real_bundle_when_available() {
    const char* bundle = std::getenv("SPEECH_POCKET_TTS_BUNDLE");
    if (!bundle || bundle[0] == '\0') {
        std::printf("  SKIP: real_bundle (SPEECH_POCKET_TTS_BUNDLE unset)\n");
        return;
    }
    const auto root = std::filesystem::path(bundle);
    speech_core::PocketTtsTokenizer tokenizer(
        (root / "vocab.json").string(), (root / "token_scores.json").string());
    assert((tokenizer.encode_ids("Hello world.") ==
            std::vector<std::int32_t>{2994, 578, 263}));
    assert((tokenizer.encode_ids("I can call your contacts,") ==
            std::vector<std::int32_t>{268, 303, 583, 312, 331, 2444, 261, 262}));
    std::printf("  PASS: real_bundle\n");
}

}  // namespace

int main() {
    std::printf("test_pocket_tts_tokenizer:\n");
    test_words_and_whitespace();
    test_unigram_scores_choose_best_path();
    test_utf8_byte_fallback();
    test_real_bundle_when_available();
    std::printf("All Pocket TTS tokenizer tests passed.\n");
    return 0;
}
