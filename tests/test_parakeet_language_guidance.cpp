#include "speech_core/models/parakeet_language_guidance.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using namespace speech_core;

void test_normalize_language_code() {
    assert(parakeet::normalize_language_code("en-US") == "en");
    assert(parakeet::normalize_language_code("FR_fr") == "fr");
    assert(parakeet::normalize_language_code("auto") == "auto");
    assert(parakeet::normalize_language_code("") == "");
    std::printf("  PASS: normalize_language_code\n");
}

void test_resolve_language_tokens() {
    const std::unordered_map<int, std::string> tokens = {
        {10, "en"},
        {11, "fr"},
        {12, "de"},
    };

    const auto resolved = parakeet::resolve_language_tokens(
        tokens, {"fr-FR", "xx", "EN_us", "fr"});

    assert(resolved.size() == 2);
    assert(resolved[0] == 11);
    assert(resolved[1] == 10);
    std::printf("  PASS: resolve_language_tokens\n");
}

void test_guidance_leaves_non_language_tokens_alone() {
    const std::unordered_map<int, std::string> tokens = {
        {10, "en"},
        {11, "fr"},
    };
    float logits[12] = {};
    logits[4] = 9.0f;
    logits[10] = 5.0f;
    logits[11] = 6.0f;
    float score = logits[4];

    const int chosen = parakeet::apply_language_guidance(
        logits, 4, &score, tokens, {10, 11});

    assert(chosen == 4);
    assert(std::abs(score - 9.0f) < 1e-6f);
    std::printf("  PASS: guidance_leaves_non_language_tokens_alone\n");
}

void test_guidance_replaces_disallowed_language_token() {
    const std::unordered_map<int, std::string> tokens = {
        {10, "en"},
        {11, "fr"},
        {12, "de"},
    };
    float logits[13] = {};
    logits[10] = 4.0f;
    logits[11] = 6.0f;
    logits[12] = 10.0f;
    float score = logits[12];

    const int chosen = parakeet::apply_language_guidance(
        logits, 12, &score, tokens, {10, 11});

    assert(chosen == 11);
    assert(std::abs(score - 6.0f) < 1e-6f);
    std::printf("  PASS: guidance_replaces_disallowed_language_token\n");
}

int main() {
    std::printf("test_parakeet_language_guidance:\n");
    test_normalize_language_code();
    test_resolve_language_tokens();
    test_guidance_leaves_non_language_tokens_alone();
    test_guidance_replaces_disallowed_language_token();
    std::printf("All Parakeet language guidance tests passed.\n");
    return 0;
}
