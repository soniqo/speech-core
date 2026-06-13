// fdb_summary CLI — thin wrapper around fdb_summary::process_dir.
// The implementation lives in fdb_summary_lib.cpp so the smoke test can
// link the same code without spawning a subprocess.

#include "fdb_summary.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void print_usage() {
    std::fprintf(stdout,
        "fdb_summary — aggregate fdb_bench JSON output into a single CSV\n"
        "\n"
        "Usage: fdb_summary --in-dir <path> --out-csv <path|->\n"
        "\n"
        "Reads every <sample_id>.json file produced by fdb_bench, buckets\n"
        "samples by (category, stt_backend, tts_backend, llm_model), and\n"
        "writes a CSV with sample count, error count, and p50/p90/p99 of\n"
        "stt/llm/tts/ttft/total timings per bucket.\n"
        "\n"
        "Options:\n"
        "  --in-dir <path>     directory of fdb_bench *.json files (required)\n"
        "  --out-csv <path|->  output CSV path; '-' for stdout (required)\n"
        "  -h, --help          this help\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string in_dir;
    std::string out_csv;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        if (f == "-h" || f == "--help") help = true;
        else if (f == "--in-dir"  && i + 1 < argc) in_dir  = argv[++i];
        else if (f == "--out-csv" && i + 1 < argc) out_csv = argv[++i];
        else {
            std::fprintf(stderr, "fdb_summary: unknown flag %s\n", f.c_str());
            print_usage();
            return 2;
        }
    }
    if (help) { print_usage(); return 0; }
    if (in_dir.empty() || out_csv.empty()) {
        std::fprintf(stderr,
            "fdb_summary: --in-dir and --out-csv are required\n");
        print_usage();
        return 2;
    }

    if (out_csv == "-") {
        return fdb_summary::process_dir(in_dir, std::cout) ? 0 : 1;
    }
    std::ofstream os(out_csv);
    if (!os) {
        std::fprintf(stderr, "fdb_summary: cannot write %s\n",
                     out_csv.c_str());
        return 1;
    }
    return fdb_summary::process_dir(in_dir, os) ? 0 : 1;
}
