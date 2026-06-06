#ifndef SPEECH_CORE_LITERT_HF_DOWNLOAD_H
#define SPEECH_CORE_LITERT_HF_DOWNLOAD_H

// Minimal, robust Hugging Face file downloader (internal to
// speech_core_models_litert). Used by sc_voxcpm2_create_from_pretrained to
// fetch a model bundle on first run, mirroring how speech-swift's
// HuggingFaceDownloader works on Apple platforms.
//
// Compiled against libcurl and only active when speech-core is built with
// SPEECH_CORE_WITH_HF_DOWNLOAD=ON. When that's off, download_bundle throws so
// the caller can surface a clear "built without download support" error.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace speech_core::hf {

/// Per-file progress. `total` is 0 when the size is unknown. Called on the
/// downloading thread; keep the callback light.
using ProgressFn = std::function<void(const std::string& file, int file_index,
                                       int file_count, uint64_t downloaded,
                                       uint64_t total)>;

/// Ensure every file in `files` exists under `dest_dir`, downloading the
/// missing ones from `https://<endpoint>/<repo>/resolve/<revision>/<file>`.
///
/// Robust against network breaks: each file streams to a `<file>.part`
/// sidecar and resumes from the existing byte offset (HTTP Range) across
/// retries with exponential backoff; the `.part` is renamed into place only
/// once the whole file is present. Already-complete files are skipped.
///
/// Creates `dest_dir` (and parents) as needed. Throws std::runtime_error on
/// unrecoverable failure (network exhausted, disk error, or — when built
/// without SPEECH_CORE_WITH_HF_DOWNLOAD — lack of download support).
void download_bundle(const std::string& repo,
                     const std::string& revision,
                     const std::vector<std::string>& files,
                     const std::string& dest_dir,
                     const ProgressFn& on_progress);

/// Whether this build has libcurl-backed download support compiled in.
bool download_supported();

}  // namespace speech_core::hf

#endif  // SPEECH_CORE_LITERT_HF_DOWNLOAD_H
