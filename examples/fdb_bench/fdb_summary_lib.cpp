// Implementation of fdb_summary::process_dir — factored out so both
// the binary (examples/fdb_bench/fdb_summary.cpp) and the smoke test
// (tests/test_fdb_summary_smoke.cpp) can link the same code without
// the test having to spawn a subprocess.

#include "fdb_summary.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct BucketKey {
    std::string category;
    std::string stt_backend;
    std::string tts_backend;
    std::string llm_model;
    bool operator<(const BucketKey& o) const {
        if (category    != o.category)    return category    < o.category;
        if (stt_backend != o.stt_backend) return stt_backend < o.stt_backend;
        if (tts_backend != o.tts_backend) return tts_backend < o.tts_backend;
        return llm_model < o.llm_model;
    }
};

struct Bucket {
    size_t sample_count = 0;
    size_t error_count  = 0;
    std::vector<long long> stt_ms;
    std::vector<long long> llm_ms;
    std::vector<long long> tts_ms;
    std::vector<long long> ttft_ms;
    std::vector<long long> total_ms;
};

// Percentile via std::nth_element on a copy. p in [0, 100]. Uses the
// "nearest-rank" method: idx = ceil(p/100 * n) - 1. So for 10 samples
// p50 -> index 4 (the 5th value), p90 -> 8, p99 -> 9. Returns 0 for an
// empty input rather than throwing — buckets that are all errors then
// surface as zero rows in the CSV without crashing.
long long percentile(std::vector<long long> v, double p) {
    if (v.empty()) return 0;
    if (p < 0.0)   p = 0.0;
    if (p > 100.0) p = 100.0;
    double rank = (p / 100.0) * static_cast<double>(v.size());
    size_t idx = (rank <= 0.0)
        ? 0
        : static_cast<size_t>(rank + (rank == std::floor(rank) ? 0.0 : 1.0)) - 1;
    if (idx >= v.size()) idx = v.size() - 1;
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

std::string get_str(const nlohmann::json& j, const char* k,
                    const std::string& def = "") {
    auto it = j.find(k);
    if (it == j.end() || !it->is_string()) return def;
    return it->get<std::string>();
}

long long get_i64(const nlohmann::json& j, const char* k,
                  long long def = 0) {
    auto it = j.find(k);
    if (it == j.end() || !it->is_number()) return def;
    return it->get<long long>();
}

}  // namespace

namespace fdb_summary {

bool process_dir(const std::string& in_dir, std::ostream& csv) {
    std::error_code ec;
    if (!fs::is_directory(in_dir, ec)) {
        std::fprintf(stderr, "fdb_summary: not a directory: %s\n",
                     in_dir.c_str());
        return false;
    }

    std::map<BucketKey, Bucket> buckets;

    for (auto& entry : fs::directory_iterator(in_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".json") continue;

        std::ifstream is(entry.path());
        if (!is) {
            std::fprintf(stderr, "fdb_summary: cannot open %s\n",
                         entry.path().string().c_str());
            continue;
        }
        nlohmann::json j;
        try {
            is >> j;
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "fdb_summary: bad json in %s: %s\n",
                         entry.path().string().c_str(), ex.what());
            continue;
        }

        BucketKey key;
        key.category    = get_str(j, "category");
        key.stt_backend = get_str(j, "stt_backend");
        key.tts_backend = get_str(j, "tts_backend");
        key.llm_model   = get_str(j, "llm_model");

        Bucket& b = buckets[key];
        b.sample_count++;
        const std::string err = get_str(j, "error");
        if (!err.empty()) {
            b.error_count++;
            continue;
        }
        auto tit = j.find("timings_ms");
        if (tit == j.end() || !tit->is_object()) continue;
        b.stt_ms  .push_back(get_i64(*tit, "stt"));
        b.llm_ms  .push_back(get_i64(*tit, "llm"));
        b.tts_ms  .push_back(get_i64(*tit, "tts"));
        b.ttft_ms .push_back(get_i64(*tit, "ttft_first_audio_from_speech_end"));
        b.total_ms.push_back(get_i64(*tit, "total_wall"));
    }

    csv << "category,stt_backend,tts_backend,llm_model,"
        << "samples,errors,"
        << "p50_stt_ms,p90_stt_ms,p99_stt_ms,"
        << "p50_llm_ms,p90_llm_ms,p99_llm_ms,"
        << "p50_tts_ms,p90_tts_ms,p99_tts_ms,"
        << "p50_ttft_ms,p90_ttft_ms,p99_ttft_ms,"
        << "p50_total_ms,p90_total_ms,p99_total_ms\n";

    for (const auto& [k, b] : buckets) {
        csv << k.category    << ','
            << k.stt_backend << ','
            << k.tts_backend << ','
            << k.llm_model   << ','
            << b.sample_count << ','
            << b.error_count  << ','
            << percentile(b.stt_ms,  50) << ','
            << percentile(b.stt_ms,  90) << ','
            << percentile(b.stt_ms,  99) << ','
            << percentile(b.llm_ms,  50) << ','
            << percentile(b.llm_ms,  90) << ','
            << percentile(b.llm_ms,  99) << ','
            << percentile(b.tts_ms,  50) << ','
            << percentile(b.tts_ms,  90) << ','
            << percentile(b.tts_ms,  99) << ','
            << percentile(b.ttft_ms, 50) << ','
            << percentile(b.ttft_ms, 90) << ','
            << percentile(b.ttft_ms, 99) << ','
            << percentile(b.total_ms, 50) << ','
            << percentile(b.total_ms, 90) << ','
            << percentile(b.total_ms, 99) << '\n';
    }
    return true;
}

}  // namespace fdb_summary
