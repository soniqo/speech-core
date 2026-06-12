#include "fdb_corpus.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fdb_bench {

namespace {

namespace fs = std::filesystem;

bool is_numeric_id(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

int64_t numeric_id_value(const std::string& s) {
    try { return std::stoll(s); }
    catch (...) { return -1; }
}

struct CategoryDef {
    FdbCategory category;
    const char* name;       // short name for CLI / JSON
    const char* dir_name;   // FDB on-disk directory
    const char* annotation; // "" if none
};

const CategoryDef kCategories[] = {
    {FdbCategory::CandorPauseHandling,    "candor_pause_handling",
        "candor_pause_handling",    "pause.json"},
    {FdbCategory::SyntheticPauseHandling, "synthetic_pause_handling",
        "synthetic_pause_handling", "pause.json"},
    {FdbCategory::SmoothTurnTaking,       "smooth_turn_taking",
        "candor_turn_taking",       "turn_taking.json"},
    {FdbCategory::UserInterruption,       "user_interruption",
        "synthetic_user_interruption", "interrupt.json"},
    {FdbCategory::Backchannel,            "backchannel",
        "icc_backchannel",          ""},
};

const CategoryDef& def_for(FdbCategory c) {
    for (const auto& d : kCategories) if (d.category == c) return d;
    return kCategories[0];
}

}  // namespace

const char* FdbCorpus::category_name(FdbCategory c) {
    return def_for(c).name;
}

const char* FdbCorpus::category_dir_name(FdbCategory c) {
    return def_for(c).dir_name;
}

std::optional<FdbCategory> FdbCorpus::parse_category(const std::string& s) {
    for (const auto& d : kCategories) {
        if (s == d.name) return d.category;
    }
    return std::nullopt;
}

std::vector<FdbSample> FdbCorpus::load(const FdbCorpusOptions& opts) {
    std::vector<FdbSample> out;
    std::error_code ec;
    if (opts.corpus_root.empty()) return out;
    if (!fs::is_directory(opts.corpus_root, ec)) return out;

    for (const auto& cd : kCategories) {
        if (opts.category && *opts.category != cd.category) continue;
        fs::path category_dir = fs::path(opts.corpus_root) / cd.dir_name;
        if (!fs::is_directory(category_dir, ec)) continue;

        std::vector<FdbSample> bucket;
        for (auto& entry : fs::directory_iterator(category_dir, ec)) {
            if (ec) break;
            if (!entry.is_directory()) continue;
            std::string sid = entry.path().filename().string();
            if (!is_numeric_id(sid)) continue;

            fs::path sd = entry.path();
            fs::path wav = sd / "input.wav";
            if (!fs::is_regular_file(wav, ec)) {
                std::fprintf(stderr,
                    "fdb_corpus: missing input.wav under %s\n",
                    sd.string().c_str());
                continue;
            }

            FdbSample s;
            s.category          = cd.category;
            s.category_dir_name = cd.dir_name;
            s.sample_id         = sid;
            s.sample_dir        = sd.string();
            s.input_wav_path    = wav.string();
            if (cd.annotation && cd.annotation[0]) {
                fs::path ann = sd / cd.annotation;
                if (fs::is_regular_file(ann, ec)) {
                    s.annotation_path = ann.string();
                }
            }
            fs::path tr = sd / "transcription.json";
            if (fs::is_regular_file(tr, ec)) {
                s.transcription_path = tr.string();
            }
            bucket.push_back(std::move(s));
        }
        std::sort(bucket.begin(), bucket.end(),
            [](const FdbSample& a, const FdbSample& b) {
                int64_t ai = numeric_id_value(a.sample_id);
                int64_t bi = numeric_id_value(b.sample_id);
                if (ai != bi) return ai < bi;
                return a.sample_id < b.sample_id;
            });
        for (auto& s : bucket) {
            out.push_back(std::move(s));
            if (opts.limit > 0 && out.size() >= opts.limit) return out;
        }
    }
    return out;
}

std::string FdbCorpus::extract_ground_truth_transcript(
    const std::string& transcription_json_path)
{
    std::ifstream is(transcription_json_path, std::ios::binary);
    if (!is) return "";
    std::stringstream ss; ss << is.rdbuf();
    std::string blob = ss.str();
    if (blob.empty()) return "";

    // Pure string scan — pull each "text":"..." occurrence and join with
    // spaces. Skip the sentinel tokens "[PAUSE]" / "[TURN-TAKING]" that
    // appear in the annotation files. The transcription.json files we
    // see in the FDB v1.0 example_data are nested arrays of {text, ...}
    // word records; this scan tolerates both that shape and the flatter
    // annotation shape without needing a JSON dep.
    std::string out;
    size_t pos = 0;
    const std::string key = "\"text\"";
    while (true) {
        size_t k = blob.find(key, pos);
        if (k == std::string::npos) break;
        k += key.size();
        // Skip whitespace + colon.
        while (k < blob.size() && (blob[k] == ' ' || blob[k] == ':' ||
                                    blob[k] == '\t')) ++k;
        if (k >= blob.size() || blob[k] != '"') { pos = k; continue; }
        ++k;
        std::string word;
        while (k < blob.size() && blob[k] != '"') {
            // Minimal escape handling — preserve escaped quote / backslash.
            if (blob[k] == '\\' && k + 1 < blob.size()) {
                word.push_back(blob[k + 1]);
                k += 2;
            } else {
                word.push_back(blob[k]);
                ++k;
            }
        }
        if (k < blob.size()) ++k;
        pos = k;
        if (word == "[PAUSE]" || word == "[TURN-TAKING]" ||
            word == "[INTERRUPT]" || word.empty()) {
            continue;
        }
        if (!out.empty()) out.push_back(' ');
        out += word;
    }
    return out;
}

}  // namespace fdb_bench
