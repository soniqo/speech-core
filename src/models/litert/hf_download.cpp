#include "hf_download.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef SPEECH_CORE_WITH_HF_DOWNLOAD
#include <chrono>
#include <thread>

#include <curl/curl.h>
#endif

namespace fs = std::filesystem;

namespace speech_core::hf {

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

int xferinfo(void* p, curl_off_t /*dltotal*/, curl_off_t dlnow,
             curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    auto* c = static_cast<ProgressCtx*>(p);
    if (c->fn && *c->fn) {
        (*c->fn)(c->file, c->idx, c->count, c->base + static_cast<uint64_t>(dlnow),
                 c->total);
    }
    return 0;  // non-zero would abort the transfer
}

// HEAD the (redirected) URL and return Content-Length, or 0 if unknown.
uint64_t remote_size(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) return 0;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
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
        const std::string url = resolve_url(repo, revision, file);

        const uint64_t total = remote_size(url);

        // Already complete? (We only rename into place once whole, so an
        // existing final_path is trusted; if we know the size, confirm it.)
        if (fs::exists(final_path)) {
            const uint64_t have = static_cast<uint64_t>(fs::file_size(final_path, ec));
            if (total == 0 || have == total) {
                if (on_progress) on_progress(file, i, file_count, have, have);
                continue;
            }
            fs::remove(final_path, ec);  // size mismatch — refetch
        }

        bool done = false;
        for (int attempt = 0; attempt < kMaxAttempts && !done; ++attempt) {
            uint64_t have = fs::exists(part_path)
                                ? static_cast<uint64_t>(fs::file_size(part_path, ec))
                                : 0;
            if (total > 0 && have > total) {
                fs::remove(part_path, ec);  // corrupt/over-long — restart
                have = 0;
            }
            if (total > 0 && have == total) {
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

            ProgressCtx pctx{&on_progress, file, i, file_count, have, total};
            fetch_once(url, part_path, have, pctx);

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
        if (on_progress) on_progress(file, i, file_count, total, total);
    }
}

#endif  // SPEECH_CORE_WITH_HF_DOWNLOAD

}  // namespace speech_core::hf
