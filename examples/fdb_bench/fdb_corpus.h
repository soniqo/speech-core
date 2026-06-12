// FDB v1.0 corpus iterator. On-disk layout (matches FDB v1.0 verbatim):
//
//   v1_0/
//   ├── candor_pause_handling/<id>/        (216 samples, 48 kHz)
//   ├── synthetic_pause_handling/<id>/     (137 samples, 16 kHz)
//   ├── candor_turn_taking/<id>/           (119 samples, 16 kHz)
//   ├── synthetic_user_interruption/<id>/  (200 samples, 16 kHz)
//   └── icc_backchannel/<id>/              ( 55 samples, 16 kHz)
//
// There is no candor/synthetic/ parent split — both subsplits live as
// sibling category directories.

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace fdb_bench {

enum class FdbCategory {
    CandorPauseHandling,
    SyntheticPauseHandling,
    SmoothTurnTaking,         // directory name: candor_turn_taking
    UserInterruption,         // directory name: synthetic_user_interruption
    Backchannel               // directory name: icc_backchannel
};

struct FdbSample {
    FdbCategory category;
    std::string category_dir_name;
    std::string sample_id;        // numeric string from the directory name
    std::string sample_dir;       // absolute path
    std::string input_wav_path;   // <sample_dir>/input.wav
    std::string annotation_path;  // category-specific *.json, "" for backchannel
    std::string transcription_path;  // transcription.json, may not exist
};

struct FdbCorpusOptions {
    std::string corpus_root;              // path to the v1_0/ directory
    std::optional<FdbCategory> category;  // nullopt = all five
    size_t limit = 0;                     // 0 = no cap
};

class FdbCorpus {
public:
    /// Enumerate samples. Sorted by (category, numeric sample id) for
    /// deterministic runs. Skips sample directories missing input.wav
    /// with a warning to stderr. Returns empty on a missing/empty
    /// corpus root.
    static std::vector<FdbSample> load(const FdbCorpusOptions& opts);

    static const char* category_name(FdbCategory c);
    static const char* category_dir_name(FdbCategory c);
    static std::optional<FdbCategory> parse_category(const std::string& s);

    /// Best-effort transcript extraction from a per-sample
    /// transcription.json file. Pure string scan — no JSON dependency.
    /// Returns "" when the file is absent or malformed. Used by the
    /// mock STT so the driver feeds a meaningful prompt to the LLM
    /// when running the bench in mock mode.
    static std::string extract_ground_truth_transcript(
        const std::string& transcription_json_path);
};

}  // namespace fdb_bench
