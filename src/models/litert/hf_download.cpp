#include "hf_download.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef SPEECH_CORE_WITH_HF_DOWNLOAD
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>
#include <thread>

#include <curl/curl.h>
#endif

namespace fs = std::filesystem;

namespace speech_core::hf {

// Pure decision logic shared by download_bundle. Compiled regardless of whether
// libcurl support is enabled, so it is unit-testable in any build.
bool final_is_complete(uint64_t final_size, uint64_t remote_size) {
    return remote_size == 0 || final_size == remote_size;
}

FetchAction plan_part_fetch(uint64_t part_size, uint64_t remote_size, uint64_t* resume_from) {
    if (remote_size > 0 && part_size > remote_size) {
        if (resume_from) *resume_from = 0;
        return FetchAction::Restart;  // over-long/corrupt .part — start over
    }
    if (remote_size > 0 && part_size == remote_size) {
        if (resume_from) *resume_from = part_size;
        return FetchAction::Complete;  // already fully downloaded
    }
    if (resume_from) *resume_from = part_size;  // resume (part_size may be 0)
    return FetchAction::Resume;
}

#ifndef SPEECH_CORE_WITH_HF_DOWNLOAD

bool download_supported() { return false; }

void download_bundle(const std::string&, const std::string&,
                     const std::vector<std::string>&, const std::string&,
                     const ProgressFn&) {
    throw std::runtime_error(
        "speech-core was built without SPEECH_CORE_WITH_HF_DOWNLOAD; cannot "
        "download model bundles. Rebuild with -DSPEECH_CORE_WITH_HF_DOWNLOAD=ON "
        "or provide the bundle directory explicitly.");
}

#else  // SPEECH_CORE_WITH_HF_DOWNLOAD

namespace {

// Default Hugging Face endpoint; overridable via HF_ENDPOINT to support
// mirrors (e.g. hf-mirror.com) without recompiling.
std::string hf_endpoint() {
    const char* e = std::getenv("HF_ENDPOINT");
    return (e && *e) ? std::string(e) : "https://huggingface.co";
}

std::string resolve_url(const std::string& repo, const std::string& revision,
                        const std::string& file) {
    return hf_endpoint() + "/" + repo + "/resolve/" + revision + "/" + file;
}

size_t write_to_file(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* f = static_cast<std::FILE*>(userdata);
    return std::fwrite(ptr, size, nmemb, f);
}

struct ProgressCtx {
    const ProgressFn* fn;
    std::string file;
    int idx;
    int count;
    uint64_t base;   // bytes already on disk before this transfer (resume offset)
    uint64_t total;  // full remote size, 0 if unknown
};

struct SegmentProgressCtx {
    std::atomic<uint64_t>* done;
    uint64_t base;
};

struct SegmentRange {
    uint64_t start;
    uint64_t end;
    fs::path path;

    uint64_t size() const { return end - start + 1; }
};

enum class ParallelFetchResult {
    Complete,
    RangeUnsupported,
    Failed,
};

int download_connections() {
    int value = 4;
    if (const char* env = std::getenv("SPEECH_CORE_DOWNLOAD_CONNECTIONS"); env && *env) {
        value = std::atoi(env);
    }
    if (value < 1) value = 1;
    if (value > 16) value = 16;
    return value;
}

uint64_t parallel_threshold_bytes() {
    return 64ull * 1024ull * 1024ull;
}

int xferinfo(void* p, curl_off_t dltotal, curl_off_t dlnow,
             curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    auto* c = static_cast<ProgressCtx*>(p);
    if (c->fn && *c->fn) {
        uint64_t total = c->total;
        if (total == 0 && dltotal > 0) {
            total = c->base + static_cast<uint64_t>(dltotal);
        }
        (*c->fn)(c->file, c->idx, c->count, c->base + static_cast<uint64_t>(dlnow),
                 total);
    }
    return 0;  // non-zero would abort the transfer
}

int segment_xferinfo(void* p, curl_off_t /*dltotal*/, curl_off_t dlnow,
                     curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    auto* c = static_cast<SegmentProgressCtx*>(p);
    if (c && c->done) {
        c->done->store(c->base + static_cast<uint64_t>(dlnow),
                       std::memory_order_relaxed);
    }
    return 0;
}

// HEAD the (redirected) URL and return Content-Length, or 0 if unknown.
uint64_t remote_size(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) return 0;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "speech-core/hf-download");
    uint64_t size = 0;
    if (curl_easy_perform(curl) == CURLE_OK) {
        curl_off_t len = 0;
        if (curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &len) == CURLE_OK &&
            len > 0) {
            size = static_cast<uint64_t>(len);
        }
    }
    curl_easy_cleanup(curl);
    return size;
}

// One GET attempt that resumes from `resume_from`. Returns true on a clean
// transfer (curl reported success); the caller validates the resulting size.
bool fetch_once(const std::string& url, const fs::path& part, uint64_t resume_from,
                ProgressCtx& pctx) {
    std::FILE* f = std::fopen(part.string().c_str(), resume_from ? "ab" : "wb");
    if (!f) throw std::runtime_error("cannot open " + part.string() + " for writing");

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::fclose(f);
        throw std::runtime_error("curl_easy_init failed");
    }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);  // HTTP >= 400 -> CURLE error
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    // Abort a stalled (not dead) socket: < 1 KiB/s for 30 s.
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 30L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "speech-core/hf-download");
    if (resume_from) {
        curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE,
                         static_cast<curl_off_t>(resume_from));
    }
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, xferinfo);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &pctx);

    CURLcode rc = curl_easy_perform(curl);
    std::fclose(f);
    if (rc != CURLE_OK) {
        std::fprintf(stderr, "[speech] download %s: %s\n", pctx.file.c_str(),
                     curl_easy_strerror(rc));
    }
    curl_easy_cleanup(curl);
    return rc == CURLE_OK;
}

bool fetch_range_once(const std::string& url, const SegmentRange& range,
                      uint64_t local_have, std::atomic<uint64_t>& done,
                      std::atomic<bool>& range_unsupported) {
    if (range_unsupported.load(std::memory_order_relaxed)) return false;
    if (local_have >= range.size()) return true;

    std::FILE* f = std::fopen(range.path.string().c_str(), local_have ? "ab" : "wb");
    if (!f) throw std::runtime_error("cannot open " + range.path.string() + " for writing");

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::fclose(f);
        throw std::runtime_error("curl_easy_init failed");
    }

    const uint64_t from = range.start + local_have;
    const std::string range_header =
        std::to_string(from) + "-" + std::to_string(range.end);
    SegmentProgressCtx pctx{&done, local_have};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 30L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "speech-core/hf-download");
    curl_easy_setopt(curl, CURLOPT_RANGE, range_header.c_str());
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, segment_xferinfo);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &pctx);

    CURLcode rc = curl_easy_perform(curl);
    long response = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response);
    std::fclose(f);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        std::fprintf(stderr, "[speech] range download %s: %s\n",
                     range.path.filename().string().c_str(), curl_easy_strerror(rc));
        return false;
    }
    if (response != 206) {
        if (response == 200) {
            range_unsupported.store(true, std::memory_order_relaxed);
        }
        std::fprintf(stderr, "[speech] range download %s: unexpected HTTP %ld\n",
                     range.path.filename().string().c_str(), response);
        return false;
    }
    return true;
}

bool fetch_segment_with_retries(const std::string& url, const SegmentRange& range,
                                std::atomic<uint64_t>& done,
                                std::atomic<bool>& range_unsupported) {
    constexpr int kMaxAttempts = 6;
    std::error_code ec;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        if (range_unsupported.load(std::memory_order_relaxed)) return false;

        uint64_t have = fs::exists(range.path)
                            ? static_cast<uint64_t>(fs::file_size(range.path, ec))
                            : 0;
        if (have > range.size()) {
            fs::remove(range.path, ec);
            have = 0;
        }
        done.store(have, std::memory_order_relaxed);
        if (have == range.size()) return true;

        if (attempt > 0) {
            int delay = 1 << (attempt - 1);
            if (delay > 16) delay = 16;
            std::fprintf(stderr, "[speech] retrying range %s (attempt %d/%d) in %ds\n",
                         range.path.filename().string().c_str(), attempt + 1,
                         kMaxAttempts, delay);
            std::this_thread::sleep_for(std::chrono::seconds(delay));
        }

        if (fetch_range_once(url, range, have, done, range_unsupported)) {
            have = fs::exists(range.path)
                       ? static_cast<uint64_t>(fs::file_size(range.path, ec))
                       : 0;
            done.store(have, std::memory_order_relaxed);
            if (have == range.size()) return true;
        }
    }
    return false;
}

ParallelFetchResult fetch_parallel_ranges(const std::string& url, const fs::path& part,
                                          uint64_t prefix_size, uint64_t total,
                                          const std::string& file, int idx, int count,
                                          const ProgressFn& on_progress) {
    const uint64_t remaining = total - prefix_size;
    int connections = download_connections();
    if (connections <= 1 || remaining < parallel_threshold_bytes()) {
        return ParallelFetchResult::Failed;
    }
    connections = static_cast<int>(
        std::min<uint64_t>(static_cast<uint64_t>(connections), remaining));

    std::vector<SegmentRange> ranges;
    ranges.reserve(static_cast<size_t>(connections));
    const uint64_t chunk = (remaining + static_cast<uint64_t>(connections) - 1) /
                           static_cast<uint64_t>(connections);
    for (int i = 0; i < connections; ++i) {
        const uint64_t start = prefix_size + static_cast<uint64_t>(i) * chunk;
        if (start >= total) break;
        const uint64_t end = std::min<uint64_t>(total - 1, start + chunk - 1);
        ranges.push_back(SegmentRange{
            start,
            end,
            fs::path(part.string() + "." + std::to_string(start) + "-" +
                     std::to_string(end) + ".seg"),
        });
    }
    if (ranges.size() <= 1) return ParallelFetchResult::Failed;

    std::vector<std::unique_ptr<std::atomic<uint64_t>>> progress;
    progress.reserve(ranges.size());
    for (const auto& range : ranges) {
        auto done = std::make_unique<std::atomic<uint64_t>>(0);
        std::error_code ec;
        if (fs::exists(range.path)) {
            uint64_t have = static_cast<uint64_t>(fs::file_size(range.path, ec));
            if (have <= range.size()) done->store(have, std::memory_order_relaxed);
        }
        progress.push_back(std::move(done));
    }

    std::atomic<int> active{static_cast<int>(ranges.size())};
    std::atomic<bool> ok{true};
    std::atomic<bool> range_unsupported{false};
    std::vector<std::thread> workers;
    workers.reserve(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        workers.emplace_back([&, i] {
            try {
                if (!fetch_segment_with_retries(url, ranges[i], *progress[i],
                                                range_unsupported)) {
                    ok.store(false, std::memory_order_relaxed);
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[speech] range worker %s: %s\n",
                             ranges[i].path.filename().string().c_str(), e.what());
                ok.store(false, std::memory_order_relaxed);
            }
            active.fetch_sub(1, std::memory_order_relaxed);
        });
    }

    while (active.load(std::memory_order_relaxed) > 0) {
        uint64_t done = prefix_size;
        for (const auto& p : progress) done += p->load(std::memory_order_relaxed);
        if (on_progress) on_progress(file, idx, count, done, total);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    for (auto& worker : workers) worker.join();

    uint64_t done = prefix_size;
    for (const auto& p : progress) done += p->load(std::memory_order_relaxed);
    if (on_progress) on_progress(file, idx, count, done, total);

    if (range_unsupported.load(std::memory_order_relaxed)) {
        return ParallelFetchResult::RangeUnsupported;
    }
    if (!ok.load(std::memory_order_relaxed)) {
        return ParallelFetchResult::Failed;
    }

    std::error_code ec;
    for (const auto& range : ranges) {
        const uint64_t have = fs::exists(range.path)
                                  ? static_cast<uint64_t>(fs::file_size(range.path, ec))
                                  : 0;
        if (have != range.size()) return ParallelFetchResult::Failed;
    }

    std::ofstream out(part, std::ios::binary | std::ios::app);
    if (!out) throw std::runtime_error("cannot open " + part.string() + " for appending");
    for (const auto& range : ranges) {
        std::ifstream in(range.path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open " + range.path.string() + " for reading");
        out << in.rdbuf();
        in.close();
        fs::remove(range.path, ec);
    }
    out.close();

    const uint64_t final_size = fs::exists(part)
                                    ? static_cast<uint64_t>(fs::file_size(part, ec))
                                    : 0;
    return final_size == total ? ParallelFetchResult::Complete
                               : ParallelFetchResult::Failed;
}

}  // namespace

bool download_supported() { return true; }

void download_bundle(const std::string& repo, const std::string& revision,
                     const std::vector<std::string>& files,
                     const std::string& dest_dir, const ProgressFn& on_progress) {
    static const bool curl_ready = [] {
        return curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
    }();
    if (!curl_ready) throw std::runtime_error("curl_global_init failed");

    std::error_code ec;
    fs::create_directories(dest_dir, ec);

    constexpr int kMaxAttempts = 6;
    const int file_count = static_cast<int>(files.size());

    for (int i = 0; i < file_count; ++i) {
        const std::string& file = files[i];
        const fs::path final_path = fs::path(dest_dir) / file;
        const fs::path part_path = fs::path(dest_dir) / (file + ".part");
        // A file may carry a subdir (e.g. "fp32-p16/foo.tflite", for the x86
        // bundle variant) — create its parent so the .part write succeeds.
        fs::create_directories(final_path.parent_path(), ec);
        const std::string url = resolve_url(repo, revision, file);

        const uint64_t total = remote_size(url);

        // Already complete? (We only rename into place once whole, so an
        // existing final_path is trusted; if we know the size, confirm it.)
        if (fs::exists(final_path)) {
            const uint64_t have = static_cast<uint64_t>(fs::file_size(final_path, ec));
            if (final_is_complete(have, total)) {
                if (on_progress) on_progress(file, i, file_count, have, have);
                continue;
            }
            fs::remove(final_path, ec);  // size mismatch — refetch
        }

        bool done = false;
        for (int attempt = 0; attempt < kMaxAttempts && !done; ++attempt) {
            const uint64_t have = fs::exists(part_path)
                                ? static_cast<uint64_t>(fs::file_size(part_path, ec))
                                : 0;
            uint64_t resume_from = 0;
            const FetchAction act = plan_part_fetch(have, total, &resume_from);
            if (act == FetchAction::Restart) {
                fs::remove(part_path, ec);  // over-long/corrupt — start over
                resume_from = 0;
            } else if (act == FetchAction::Complete) {
                done = true;
                break;
            }
            if (attempt > 0) {
                // Exponential backoff: 1s, 2s, 4s, … capped at 16s.
                int delay = 1 << (attempt - 1);
                if (delay > 16) delay = 16;
                std::fprintf(stderr, "[speech] retrying %s (attempt %d/%d) in %ds\n",
                             file.c_str(), attempt + 1, kMaxAttempts, delay);
                std::this_thread::sleep_for(std::chrono::seconds(delay));
            }

            if (total > 0 && download_connections() > 1 &&
                (total - resume_from) >= parallel_threshold_bytes()) {
                std::fprintf(stderr,
                             "[speech] downloading %s with %d parallel ranges\n",
                             file.c_str(), download_connections());
                const ParallelFetchResult parallel = fetch_parallel_ranges(
                    url, part_path, resume_from, total, file, i, file_count,
                    on_progress);
                if (parallel == ParallelFetchResult::Complete) {
                    done = true;
                } else if (parallel == ParallelFetchResult::RangeUnsupported) {
                    std::fprintf(stderr,
                                 "[speech] range download unsupported for %s; "
                                 "falling back to single stream\n",
                                 file.c_str());
                    ProgressCtx pctx{&on_progress, file, i, file_count,
                                     resume_from, total};
                    fetch_once(url, part_path, resume_from, pctx);
                }
            } else {
                ProgressCtx pctx{&on_progress, file, i, file_count, resume_from, total};
                fetch_once(url, part_path, resume_from, pctx);
            }

            const uint64_t now = fs::exists(part_path)
                                     ? static_cast<uint64_t>(fs::file_size(part_path, ec))
                                     : 0;
            done = (total > 0) ? (now == total) : (now > 0);
        }

        if (!done) {
            throw std::runtime_error("failed to download " + file + " from " + repo +
                                     " after " + std::to_string(kMaxAttempts) +
                                     " attempts");
        }

        fs::remove(final_path, ec);  // no-op if absent
        fs::rename(part_path, final_path, ec);
        if (ec) {
            throw std::runtime_error("could not finalize " + final_path.string() +
                                     ": " + ec.message());
        }
        const uint64_t final_size = total > 0
                                      ? total
                                      : static_cast<uint64_t>(fs::file_size(final_path, ec));
        if (on_progress) on_progress(file, i, file_count, final_size, final_size);
    }
}

#endif  // SPEECH_CORE_WITH_HF_DOWNLOAD

}  // namespace speech_core::hf
