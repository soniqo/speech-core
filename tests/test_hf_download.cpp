// Unit tests for the Hugging Face downloader's pure decision logic — the
// resume / skip / restart rules behind sc_voxcpm2_create_from_pretrained. The
// libcurl transport is exercised separately (end-to-end); this guards OUR
// decisions, which is where the bugs would live. No network or libcurl needed:
// hf_download.cpp's pure helpers compile regardless of the build flag.

#include "hf_download.h"

#include <cstdint>
#include <cstdio>

using speech_core::hf::FetchAction;
using speech_core::hf::final_is_complete;
using speech_core::hf::plan_part_fetch;

static int g_fail = 0;
#define CHECK(cond, msg)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL: %s\n", (msg));    \
            ++g_fail;                                     \
        }                                                 \
    } while (0)

int main() {
    // --- final_is_complete: an existing final file is trusted when the size
    //     matches or the remote size is unknown (0). ---
    CHECK(final_is_complete(100, 100), "exact size match is complete");
    CHECK(final_is_complete(100, 0), "unknown remote size trusts existing final");
    CHECK(!final_is_complete(99, 100), "short final is incomplete");
    CHECK(!final_is_complete(101, 100), "over-long final is incomplete");

    // --- plan_part_fetch: resume / complete / restart on the .part file. ---
    uint64_t off = 999;  // sentinel; must be overwritten

    CHECK(plan_part_fetch(0, 1000, &off) == FetchAction::Resume && off == 0,
          "no bytes yet -> resume from 0");

    off = 999;
    CHECK(plan_part_fetch(400, 1000, &off) == FetchAction::Resume && off == 400,
          "partial < remote -> resume from offset");

    off = 999;
    CHECK(plan_part_fetch(1000, 1000, &off) == FetchAction::Complete && off == 1000,
          "partial == remote -> complete");

    off = 999;
    CHECK(plan_part_fetch(1500, 1000, &off) == FetchAction::Restart && off == 0,
          "over-long .part -> restart from 0");

    off = 999;
    CHECK(plan_part_fetch(700, 0, &off) == FetchAction::Resume && off == 700,
          "unknown remote size -> resume from what we have");

    off = 999;
    CHECK(plan_part_fetch(0, 0, &off) == FetchAction::Resume && off == 0,
          "unknown remote size, nothing local -> resume from 0");

    // resume_from may be null.
    CHECK(plan_part_fetch(50, 100, nullptr) == FetchAction::Resume,
          "null resume_from is tolerated");

    if (g_fail == 0) {
        std::fprintf(stderr, "test_hf_download: all assertions passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_hf_download: %d failure(s)\n", g_fail);
    return 1;
}
