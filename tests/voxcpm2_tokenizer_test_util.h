#pragma once

#include "speech_core/models/voxcpm2_prompt.h"
#include "speech_core/models/voxcpm2_tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

namespace speech_core_test {

inline void print_ids(const std::vector<int>& ids) {
    std::printf("[");
    for (size_t i = 0; i < ids.size(); ++i) {
        std::printf("%s%d", i ? "," : "", ids[i]);
    }
    std::printf("]");
}

inline bool run_voxcpm2_tokenizer_reference_cases(const std::string& tok_path) {
    std::printf("  test_voxcpm2_tokenizer ... ");

    if (speech_core::format_voxcpm2_prompt("hello", "") != "hello" ||
        speech_core::format_voxcpm2_prompt("hello", "calm") != "(calm)hello") {
        std::printf("bad prompt formatting ");
        return false;
    }

    speech_core::VoxCPM2Tokenizer t(tok_path);
    if (t.bos_id() != 1 || t.eos_id() != 2 || t.unk_id() != 0 || t.vocab_size() <= 73000) {
        std::printf("bad tokenizer metadata ");
        return false;
    }

    // Reference outputs from upstream VoxCPM2's text tokenizer wrapper:
    // `mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(...))`.
    // Chinese and Japanese Han cases intentionally differ from raw HuggingFace
    // tokenizer IDs: VoxCPM2 splits merged multi-character Han tokens back to
    // single characters before ID conversion.
    struct Case { std::string text; std::vector<int> ids; };
    const std::vector<Case> cases = {
        {"Hello world",                    {1, 21045, 2809}},
        {"The quick brown fox jumps over the lazy dog.",
                                           {1, 1507, 4766, 13329, 49712, 43384, 1865, 1358, 29117, 6595, 72}},
        {"Hello, world!",                  {1, 21045, 59342, 2809, 73}},
        {"\xE4\xBD\xA0\xE5\xA5\xBD",       {1, 59320, 59496, 59495}},
        {"\xE8\xAF\xB7\xE7\xA1\xAE\xE8\xAE\xA4\xE4\xBA\x91\xE7\xAB\xAF\xE8\xAF\xAD\xE9\x9F\xB3\xE5\x90\x88\xE6\x88\x90\xE4\xBB\x8A\xE5\xA4\xA9\xE6\xB8\x85\xE6\x99\xB0\xE7\xA8\xB3\xE5\xAE\x9A\xE3\x80\x82",
                                           {1, 32435, 59699, 59638, 59968, 60197, 59836, 59941, 59474, 59452, 59856, 59534, 59726, 61533, 60334, 59421, 66}},
        {"\xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4\x20\xEC\x9D\x8C\xEC\x84\xB1\x20\xED\x95\xA9\xEC\x84\xB1\xEC\x9D\x84\x20\xED\x99\x95\xEC\x9D\xB8\xED\x95\xA9\xEB\x8B\x88\xEB\x8B\xA4\x2E",
                                           {1, 59320, 62385, 1323, 1270, 1262, 62793, 59320, 1325, 1246, 1229, 63409, 59320, 63966, 63409, 62274, 59320, 1326, 1242, 1238, 62946, 63966, 35773, 72}},
        {"\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF\xE4\xB8\x96\xE7\x95\x8C\xE3\x80\x82",
                                           {1, 59320, 62190, 62524, 61377, 63251, 61579, 59792, 59868, 66}},
        {"\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x81\xAE\xE9\x9F\xB3\xE5\xA3\xB0\xE5\x90\x88\xE6\x88\x90\xE3\x82\x92\xE7\xA2\xBA\xE8\xAA\x8D\xE3\x81\x97\xE3\x81\xBE\xE3\x81\x99\xE3\x80\x82",
                                           {1, 59420, 59416, 65419, 60931, 59941, 60068, 59474, 59452, 61417, 65013, 72626, 61432, 32902, 66}},
        {"caf\xC3\xA9",                    {1, 33903, 60025}},
        {"",                               {1}},
        {" ",                              {1, 1345}},
        {"a b  c",                         {1, 1348, 1376, 1345, 59333}},
        {"\xF0\xA0\xAE\xB7",               {1, 59320, 1329, 1249, 1263, 1272}},
    };

    for (const auto& c : cases) {
        auto got = t.encode(c.text);
        if (got != c.ids) {
            std::printf("\n    encode(\"%s\"): got ", c.text.c_str());
            print_ids(got);
            std::printf(", want ");
            print_ids(c.ids);
            std::printf(" ");
            return false;
        }
        const std::string decoded = t.decode(got);
        if (decoded != c.text) {
            std::printf("\n    decode(encode(\"%s\")) -> \"%s\" ", c.text.c_str(), decoded.c_str());
            return false;
        }
    }

    std::printf("ok\n");
    return true;
}

}  // namespace speech_core_test
