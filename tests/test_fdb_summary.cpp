// Smoke tests for the M3 fdb_summary post-processor. Writes canned
// per-sample JSON files to a temp dir, calls process_dir() directly
// (linked from fdb_summary_lib.cpp), and parses the resulting CSV
// to verify the header + per-bucket percentiles.

// Force asserts on even under RelWithDebInfo / sanitizer builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "fdb_summary.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CsvRow {
    std::vector<std::string> cells;
    const std::string& at(size_t i) const { return cells.at(i); }
    long long ll(size_t i) const { return std::stoll(cells.at(i)); }
};

std::vector<CsvRow> parse_csv(const std::string& text) {
    std::vector<CsvRow> rows;
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty()) continue;
        CsvRow r;
        std::string cell;
        std::istringstream ls(line);
        while (std::getline(ls, cell, ',')) r.cells.push_back(cell);
        rows.push_back(std::move(r));
    }
    return rows;
}

void write_sample(const fs::path& dir, const std::string& id,
                  const std::string& category,
                  const std::string& stt, const std::string& tts,
                  const std::string& llm_model,
                  long long stt_ms, long long llm_ms, long long tts_ms,
                  long long ttft_ms, long long total_ms,
                  const std::string& error = "")
{
    fs::create_directories(dir);
    std::ofstream os(dir / (id + ".json"));
    os << "{\n"
       << "  \"sample_id\": \"" << id << "\",\n"
       << "  \"category\": \"" << category << "\",\n"
       << "  \"category_dir\": \"" << category << "\",\n"
       << "  \"input_wav\": \"/tmp/x.wav\",\n"
       << "  \"input_duration_sec\": 1.0,\n"
       << "  \"ground_truth_transcript\": \"hi\",\n"
       << "  \"agent_transcript_input\": \"hi\",\n"
       << "  \"output_wav\": \"" << id << ".wav\",\n"
       << "  \"output_duration_sec\": 0.5,\n"
       << "  \"output_sample_rate\": 24000,\n"
       << "  \"timings_ms\": {\n"
       << "    \"stt\": " << stt_ms << ",\n"
       << "    \"llm\": " << llm_ms << ",\n"
       << "    \"tts\": " << tts_ms << ",\n"
       << "    \"ttft_first_audio_from_speech_end\": " << ttft_ms << ",\n"
       << "    \"total_wall\": " << total_ms << "\n"
       << "  },\n"
       << "  \"stt_backend\": \"" << stt << "\",\n"
       << "  \"tts_backend\": \"" << tts << "\",\n"
       << "  \"llm_model\": \"" << llm_model << "\",\n"
       << "  \"error\": \"" << error << "\"\n"
       << "}\n";
}

fs::path scratch_dir() {
    static auto suffix = std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return fs::temp_directory_path() / ("fdb_summary_smoke_" + suffix);
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

void test_header_matches_schema() {
    auto dir = scratch_dir() / "header";
    fs::create_directories(dir);
    write_sample(dir, "1", "candor_pause_handling", "mock", "mock",
                 "llama3.2:1b", 10, 20, 30, 40, 50);

    std::ostringstream csv;
    assert(fdb_summary::process_dir(dir.string(), csv));
    auto rows = parse_csv(csv.str());
    assert(rows.size() == 2);  // header + one bucket row

    const auto& h = rows[0].cells;
    assert(h.size() == 21);
    assert(h[0]  == "category");
    assert(h[1]  == "stt_backend");
    assert(h[2]  == "tts_backend");
    assert(h[3]  == "llm_model");
    assert(h[4]  == "samples");
    assert(h[5]  == "errors");
    assert(h[6]  == "p50_stt_ms");
    assert(h[7]  == "p90_stt_ms");
    assert(h[8]  == "p99_stt_ms");
    assert(h[20] == "p99_total_ms");

    fs::remove_all(dir);
    std::printf("  PASS: header_matches_schema\n");
}

void test_single_sample_bucket_degenerate() {
    auto dir = scratch_dir() / "single";
    fs::create_directories(dir);
    write_sample(dir, "1", "smooth_turn_taking", "mock", "mock",
                 "llama3.2:1b", 7, 13, 17, 19, 23);
    std::ostringstream csv;
    assert(fdb_summary::process_dir(dir.string(), csv));
    auto rows = parse_csv(csv.str());
    assert(rows.size() == 2);
    const auto& r = rows[1];
    assert(r.at(0) == "smooth_turn_taking");
    assert(r.at(4) == "1");  // sample_count
    assert(r.at(5) == "0");  // error_count
    // Every percentile collapses to the single value.
    assert(r.ll(6) == 7);   assert(r.ll(7) == 7);   assert(r.ll(8) == 7);
    assert(r.ll(9) == 13);  assert(r.ll(10) == 13); assert(r.ll(11) == 13);
    assert(r.ll(12) == 17); assert(r.ll(13) == 17); assert(r.ll(14) == 17);
    assert(r.ll(15) == 19); assert(r.ll(16) == 19); assert(r.ll(17) == 19);
    assert(r.ll(18) == 23); assert(r.ll(19) == 23); assert(r.ll(20) == 23);
    fs::remove_all(dir);
    std::printf("  PASS: single_sample_bucket_degenerate\n");
}

void test_percentiles_over_ten_samples() {
    auto dir = scratch_dir() / "ten";
    fs::create_directories(dir);
    // 10 samples in one bucket with total_wall = 10, 20, 30 ... 100.
    // p50 should be 50, p90 = 90, p99 = 100.
    for (int i = 1; i <= 10; ++i) {
        write_sample(dir, std::to_string(i),
                     "candor_pause_handling", "mock", "mock", "m",
                     /*stt*/ i * 1, /*llm*/ i * 2, /*tts*/ i * 3,
                     /*ttft*/ i * 5, /*total*/ i * 10);
    }
    std::ostringstream csv;
    assert(fdb_summary::process_dir(dir.string(), csv));
    auto rows = parse_csv(csv.str());
    assert(rows.size() == 2);
    const auto& r = rows[1];
    assert(r.at(4) == "10");
    assert(r.at(5) == "0");
    // total_wall percentiles (positions 18, 19, 20)
    assert(r.ll(18) == 50);
    assert(r.ll(19) == 90);
    assert(r.ll(20) == 100);
    // stt percentiles (6, 7, 8): values 1..10 → p50=5, p90=9, p99=10
    assert(r.ll(6) == 5);
    assert(r.ll(7) == 9);
    assert(r.ll(8) == 10);
    fs::remove_all(dir);
    std::printf("  PASS: percentiles_over_ten_samples\n");
}

void test_all_errors_bucket() {
    auto dir = scratch_dir() / "errors";
    fs::create_directories(dir);
    for (int i = 0; i < 3; ++i) {
        write_sample(dir, std::to_string(i),
                     "user_interruption", "mock", "mock", "m",
                     0, 0, 0, 0, 0,
                     "LLM failed: connection refused");
    }
    std::ostringstream csv;
    assert(fdb_summary::process_dir(dir.string(), csv));
    auto rows = parse_csv(csv.str());
    assert(rows.size() == 2);
    const auto& r = rows[1];
    assert(r.at(4) == "3");
    assert(r.at(5) == "3");
    // All percentile cells should be 0 (no successful samples).
    for (size_t i = 6; i <= 20; ++i) assert(r.ll(i) == 0);
    fs::remove_all(dir);
    std::printf("  PASS: all_errors_bucket\n");
}

void test_multiple_buckets_sorted() {
    auto dir = scratch_dir() / "multi";
    fs::create_directories(dir);
    // Two buckets distinguished by category.
    write_sample(dir, "1", "smooth_turn_taking", "mock", "mock", "m",
                 1, 1, 1, 1, 100);
    write_sample(dir, "2", "candor_pause_handling", "mock", "mock", "m",
                 1, 1, 1, 1, 200);
    write_sample(dir, "3", "candor_pause_handling", "mock", "mock", "m",
                 1, 1, 1, 1, 300);

    std::ostringstream csv;
    assert(fdb_summary::process_dir(dir.string(), csv));
    auto rows = parse_csv(csv.str());
    assert(rows.size() == 3);  // header + 2 buckets
    // std::map orders buckets by (category, ...); "candor..." < "smooth..."
    assert(rows[1].at(0) == "candor_pause_handling");
    assert(rows[1].at(4) == "2");
    assert(rows[2].at(0) == "smooth_turn_taking");
    assert(rows[2].at(4) == "1");
    fs::remove_all(dir);
    std::printf("  PASS: multiple_buckets_sorted\n");
}

void test_skips_unparseable_files() {
    auto dir = scratch_dir() / "bad";
    fs::create_directories(dir);
    write_sample(dir, "1", "candor_pause_handling", "mock", "mock", "m",
                 5, 5, 5, 5, 50);
    // Write a junk file alongside.
    {
        std::ofstream os(dir / "junk.json");
        os << "this is not json {";
    }
    std::ostringstream csv;
    assert(fdb_summary::process_dir(dir.string(), csv));
    auto rows = parse_csv(csv.str());
    assert(rows.size() == 2);
    assert(rows[1].at(4) == "1");
    fs::remove_all(dir);
    std::printf("  PASS: skips_unparseable_files\n");
}

void test_missing_dir_returns_false() {
    std::ostringstream csv;
    assert(!fdb_summary::process_dir(
        "/this/path/should/not/exist/anywhere", csv));
    std::printf("  PASS: missing_dir_returns_false\n");
}

}  // namespace

int main() {
    std::printf("test_fdb_summary:\n");
    test_header_matches_schema();
    test_single_sample_bucket_degenerate();
    test_percentiles_over_ten_samples();
    test_all_errors_bucket();
    test_multiple_buckets_sorted();
    test_skips_unparseable_files();
    test_missing_dir_returns_false();
    std::printf("All fdb_summary smoke tests passed.\n");
    return 0;
}
