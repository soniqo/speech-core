// Debug CLI for the Hugging Face downloader behind sc_voxcpm2_create_from_pretrained.
// Exercises speech_core::hf::download_bundle directly so resume/retry behaviour
// can be tested without pulling a full model bundle.
//
// Usage: hf_fetch <repo> <revision> <dest_dir> <file> [file...]
//   e.g. hf_fetch soniqo/VoxCPM2-LiteRT main /tmp/vox tokenizer.json
//
// Only built when SPEECH_CORE_WITH_HF_DOWNLOAD=ON.

#include "hf_download.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s <repo> <revision> <dest_dir> <file> [file...]\n",
                     argv[0]);
        return 2;
    }
    const std::string repo = argv[1];
    const std::string revision = argv[2];
    const std::string dest = argv[3];
    std::vector<std::string> files;
    for (int i = 4; i < argc; ++i) files.emplace_back(argv[i]);

    try {
        speech_core::hf::download_bundle(
            repo, revision, files, dest,
            [](const std::string& f, int idx, int cnt, uint64_t done, uint64_t total) {
                double pct = total ? (100.0 * static_cast<double>(done) /
                                      static_cast<double>(total))
                                   : 0.0;
                std::fprintf(stderr, "\r[%d/%d] %s %llu/%llu (%.1f%%)        ",
                             idx + 1, cnt, f.c_str(),
                             static_cast<unsigned long long>(done),
                             static_cast<unsigned long long>(total), pct);
                std::fflush(stderr);
            });
        std::fprintf(stderr, "\nOK\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
