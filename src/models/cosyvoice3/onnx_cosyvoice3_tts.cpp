#include "speech_core/models/onnx_cosyvoice3_tts.h"

#include "speech_core/models/onnx_engine.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace speech_core {

namespace {

using Clock = std::chrono::steady_clock;

int64_t elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - t0).count();
}

struct OrtStringHandle {
    const OrtApi* api = nullptr;
    OrtAllocator* alloc = nullptr;
    char* p = nullptr;
    OrtStringHandle(const OrtApi* a, OrtAllocator* al) : api(a), alloc(al) {}
    ~OrtStringHandle() {
        if (p && api && alloc) api->AllocatorFree(alloc, p);
    }
    OrtStringHandle(const OrtStringHandle&) = delete;
    OrtStringHandle& operator=(const OrtStringHandle&) = delete;
};

std::string utf8_codepoint(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

std::vector<std::string> bytes_to_unicode() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::vector<std::string> out(256);
    for (size_t i = 0; i < bs.size(); ++i) {
        out[static_cast<size_t>(bs[i])] = utf8_codepoint(static_cast<uint32_t>(cs[i]));
    }
    return out;
}

bool is_ascii_letter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool is_ascii_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

bool is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool is_non_ascii_lead(unsigned char c) {
    return (c & 0xC0) != 0x80;
}

size_t utf8_next(const std::string& s, size_t i) {
    if (i >= s.size()) return i;
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = 1;
    if ((c & 0xE0) == 0xC0) n = 2;
    else if ((c & 0xF0) == 0xE0) n = 3;
    else if ((c & 0xF8) == 0xF0) n = 4;
    return std::min(s.size(), i + n);
}

class QwenBpeTokenizer {
public:
    explicit QwenBpeTokenizer(const std::string& dir) {
        byte_encoder_ = bytes_to_unicode();
        load_vocab(dir + "/vocab.json");
        load_merges(dir + "/merges.txt");
        add_cosyvoice3_specials();
    }

    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> out;
        size_t i = 0;
        while (i < text.size()) {
            int special_id = -1;
            size_t special_len = 0;
            for (const auto& kv : special_ids_) {
                const std::string& tok = kv.first;
                if (tok.size() > special_len
                    && i + tok.size() <= text.size()
                    && text.compare(i, tok.size(), tok) == 0) {
                    special_id = kv.second;
                    special_len = tok.size();
                }
            }
            if (special_id >= 0) {
                out.push_back(special_id);
                i += special_len;
                continue;
            }

            size_t next_special = text.size();
            for (const auto& kv : special_ids_) {
                size_t p = text.find(kv.first, i);
                if (p != std::string::npos) next_special = std::min(next_special, p);
            }
            auto pieces = pretokenize(text.substr(i, next_special - i));
            for (const auto& piece : pieces) {
                auto ids = bpe(byte_encode(piece));
                out.insert(out.end(), ids.begin(), ids.end());
            }
            i = next_special;
        }
        return out;
    }

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> merge_rank_;
    std::unordered_map<std::string, int> special_ids_;
    std::vector<std::string> byte_encoder_;

    void load_vocab(const std::string& path) {
        const std::string text = ::json::read_file(path);
        if (text.empty()) throw std::runtime_error("CosyVoice3 tokenizer: cannot read " + path);
        size_t i = 0;
        ::json::skip_ws(text, i);
        if (i >= text.size() || text[i] != '{') {
            throw std::runtime_error("CosyVoice3 tokenizer: vocab.json is not an object");
        }
        ++i;
        while (i < text.size()) {
            ::json::skip_ws(text, i);
            if (i >= text.size() || text[i] == '}') break;
            if (text[i] == ',') { ++i; continue; }
            std::string tok = ::json::parse_string(text, i);
            ::json::skip_ws(text, i);
            if (i < text.size() && text[i] == ':') ++i;
            std::string val = ::json::parse_value_raw(text, i);
            if (!tok.empty() && !val.empty()) vocab_[tok] = std::stoi(val);
        }
    }

    void load_merges(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("CosyVoice3 tokenizer: cannot read " + path);
        std::string line;
        int rank = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            merge_rank_[line] = rank++;
        }
    }

    void add_cosyvoice3_specials() {
        const int base = static_cast<int>(vocab_.size());
        const std::vector<std::string> specials = {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|endofprompt|>", "[breath]", "<strong>", "</strong>",
            "[noise]", "[laughter]", "[cough]", "[clucking]", "[accent]",
            "[quick_breath]", "<laughter>", "</laughter>", "[hissing]",
            "[sigh]", "[vocalized-noise]", "[lipsmack]", "[mn]",
            "<|endofsystem|>",
        };
        for (size_t i = 0; i < specials.size(); ++i) {
            special_ids_[specials[i]] = base + static_cast<int>(i);
        }
    }

    static std::vector<std::string> pretokenize(const std::string& s) {
        std::vector<std::string> out;
        size_t i = 0;
        while (i < s.size()) {
            size_t start = i;
            std::string prefix;
            if (is_space(static_cast<unsigned char>(s[i]))) {
                size_t j = i;
                while (j < s.size() && is_space(static_cast<unsigned char>(s[j]))) ++j;
                if (j == s.size()) {
                    out.push_back(s.substr(i));
                    break;
                }
                prefix = s.substr(i, j - i);
                i = j;
                start = i;
            }

            unsigned char c = static_cast<unsigned char>(s[i]);
            enum class Kind { Letter, Digit, Punct };
            Kind kind = Kind::Punct;
            if (is_ascii_letter(c) || c >= 0x80) kind = Kind::Letter;
            else if (is_ascii_digit(c)) kind = Kind::Digit;

            size_t j = i;
            while (j < s.size()) {
                unsigned char d = static_cast<unsigned char>(s[j]);
                if (is_space(d)) break;
                bool same = false;
                if (kind == Kind::Letter) same = is_ascii_letter(d) || d >= 0x80;
                else if (kind == Kind::Digit) same = is_ascii_digit(d);
                else same = !is_ascii_letter(d) && !is_ascii_digit(d) && d < 0x80;
                if (!same) break;
                j = (d >= 0x80 && is_non_ascii_lead(d)) ? utf8_next(s, j) : j + 1;
            }
            out.push_back(prefix + s.substr(start, j - start));
            i = j;
        }
        return out;
    }

    std::vector<std::string> byte_encode(const std::string& s) const {
        std::vector<std::string> out;
        out.reserve(s.size());
        for (unsigned char b : s) out.push_back(byte_encoder_[b]);
        return out;
    }

    std::vector<int64_t> bpe(std::vector<std::string> tokens) const {
        while (tokens.size() >= 2) {
            int best = std::numeric_limits<int>::max();
            long idx = -1;
            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                std::string key = tokens[i] + ' ' + tokens[i + 1];
                auto it = merge_rank_.find(key);
                if (it != merge_rank_.end() && it->second < best) {
                    best = it->second;
                    idx = static_cast<long>(i);
                }
            }
            if (idx < 0) break;
            tokens[static_cast<size_t>(idx)] += tokens[static_cast<size_t>(idx + 1)];
            tokens.erase(tokens.begin() + idx + 1);
        }
        std::vector<int64_t> ids;
        ids.reserve(tokens.size());
        for (const auto& t : tokens) {
            auto it = vocab_.find(t);
            if (it == vocab_.end()) {
                throw std::runtime_error("CosyVoice3 tokenizer: token missing from vocab");
            }
            ids.push_back(it->second);
        }
        return ids;
    }
};

void write_u32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

uint32_t read_u32(const uint8_t*& p, const uint8_t* end) {
    if (end - p < 4) throw std::runtime_error("CosyVoice3 conditioning blob truncated");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
    p += 4;
    return v;
}

template <typename T>
void write_vec(std::vector<uint8_t>& out, const std::vector<T>& v) {
    write_u32(out, static_cast<uint32_t>(v.size()));
    const uint8_t* p = reinterpret_cast<const uint8_t*>(v.data());
    out.insert(out.end(), p, p + v.size() * sizeof(T));
}

template <typename T>
std::vector<T> read_vec(const uint8_t*& p, const uint8_t* end) {
    const uint32_t n = read_u32(p, end);
    const size_t bytes = static_cast<size_t>(n) * sizeof(T);
    if (static_cast<size_t>(end - p) < bytes) {
        throw std::runtime_error("CosyVoice3 conditioning blob truncated");
    }
    std::vector<T> out(n);
    std::memcpy(out.data(), p, bytes);
    p += bytes;
    return out;
}

}  // namespace

struct OnnxCosyVoice3Tts::Impl {
    explicit Impl(const std::string& bundle_dir, bool hw_accel) {
        auto& engine = OnnxEngine::get();
        api = engine.api();
        mem = engine.cpu_memory();

        tokenizer = std::make_unique<QwenBpeTokenizer>(bundle_dir + "/CosyVoice-BlankEN");
        prefill = engine.load(bundle_dir + "/llm_prefill.onnx", hw_accel, true);
        step = engine.load(bundle_dir + "/llm_step.onnx", hw_accel, true);
        flow_frontend = engine.load(bundle_dir + "/flow_frontend.onnx", hw_accel);
        flow_estimator = engine.load(bundle_dir + "/flow.decoder.estimator.fp32.onnx", hw_accel);
        hift = engine.load(bundle_dir + "/hift.onnx", hw_accel);
        hift_128 = load_optional(engine, bundle_dir + "/hift_128.onnx", hw_accel);
        hift_256 = load_optional(engine, bundle_dir + "/hift_256.onnx", hw_accel);
    }

    ~Impl() {
        if (api) {
            if (hift_256) api->ReleaseSession(hift_256);
            if (hift_128) api->ReleaseSession(hift_128);
            if (hift) api->ReleaseSession(hift);
            if (flow_estimator) api->ReleaseSession(flow_estimator);
            if (flow_frontend) api->ReleaseSession(flow_frontend);
            if (step) api->ReleaseSession(step);
            if (prefill) api->ReleaseSession(prefill);
        }
    }

    const OrtApi* api = nullptr;
    OrtMemoryInfo* mem = nullptr;
    OrtSession* prefill = nullptr;
    OrtSession* step = nullptr;
    OrtSession* flow_frontend = nullptr;
    OrtSession* flow_estimator = nullptr;
    OrtSession* hift = nullptr;
    OrtSession* hift_128 = nullptr;
    OrtSession* hift_256 = nullptr;
    std::unique_ptr<QwenBpeTokenizer> tokenizer;
    Conditioning conditioning;
    bool has_conditioning = false;
    std::atomic<bool> cancelled{false};

    static constexpr int kTextSlots = 192;
    static constexpr int kPromptSpeechSlots = 320;
    static constexpr int kCacheSlots = 1536;
    static constexpr int kLayers = 24;
    static constexpr int kKvFloats = 1 * 2 * kCacheSlots * 64;
    static constexpr int kSpeechTokenSize = 6561;
    static constexpr int kSpeechTokenExtra = 200;
    static constexpr int kFlowTokenSlots = 512;
    static constexpr int kMelSlots = 1024;
    static constexpr int kHiftFrames128 = 128;
    static constexpr int kHiftFrames256 = 256;
    static constexpr int kHiftFrames = 512;

    OrtSession* load_optional(OnnxEngine& engine,
                              const std::string& path,
                              bool hw_accel) {
        std::ifstream f(path);
        if (!f.good()) return nullptr;
        return engine.load(path, hw_accel);
    }

    OrtValue* make_i64(const int64_t* data, size_t n, const int64_t* shape, size_t rank) {
        OrtValue* v = nullptr;
        ort_check(api, api->CreateTensorWithDataAsOrtValue(
            mem, const_cast<int64_t*>(data), n * sizeof(int64_t),
            shape, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &v));
        return v;
    }

    OrtValue* make_f32(const float* data, size_t n, const int64_t* shape, size_t rank) {
        OrtValue* v = nullptr;
        ort_check(api, api->CreateTensorWithDataAsOrtValue(
            mem, const_cast<float*>(data), n * sizeof(float),
            shape, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v));
        return v;
    }

    std::vector<float> copy_f32(OrtValue* v, size_t n) {
        float* p = nullptr;
        ort_check(api, api->GetTensorMutableData(v, reinterpret_cast<void**>(&p)));
        return std::vector<float>(p, p + n);
    }

    std::vector<int64_t> encode_text(const std::string& text) const {
        return tokenizer->encode(text);
    }

    int sample_token(const std::vector<float>& logits,
                     const std::vector<int64_t>& generated,
                     int min_tokens,
                     std::mt19937& rng,
                     int top_k,
                     double top_p) const {
        std::vector<float> scores = logits;
        const int vocab_size = static_cast<int>(scores.size());
        const int suppress_end = std::min(vocab_size,
                                          kSpeechTokenSize + kSpeechTokenExtra);
        // CosyVoice reserves speech_token_size..speech_token_size+2 as stop
        // tokens. The remaining extra rows are padding/post-stop ids and should
        // never be sampled; keeping them live makes generation run to max cap.
        for (int i = kSpeechTokenSize + 3; i < suppress_end; ++i) {
            scores[static_cast<size_t>(i)] =
                -std::numeric_limits<float>::infinity();
        }
        if (static_cast<int>(generated.size()) < min_tokens) {
            const int stop_end = std::min(vocab_size, kSpeechTokenSize + 3);
            for (int i = kSpeechTokenSize; i < stop_end; ++i) {
                scores[static_cast<size_t>(i)] = -std::numeric_limits<float>::infinity();
            }
        }

        auto draw_from_scores = [&](const std::vector<float>& raw,
                                    bool nucleus) -> int {
            float max_v = -std::numeric_limits<float>::infinity();
            for (float v : raw) max_v = std::max(max_v, v);
            std::vector<std::pair<float, int>> probs;
            probs.reserve(raw.size());
            double total = 0.0;
            for (int i = 0; i < static_cast<int>(raw.size()); ++i) {
                if (!std::isfinite(raw[static_cast<size_t>(i)])) continue;
                const float p = std::exp(raw[static_cast<size_t>(i)] - max_v);
                probs.push_back({p, i});
                total += p;
            }
            if (probs.empty() || total <= 0.0) return 0;
            for (auto& p : probs) p.first = static_cast<float>(p.first / total);
            if (nucleus) {
                std::stable_sort(probs.begin(), probs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                double cum = 0.0;
                size_t keep = 0;
                while (keep < probs.size() && keep < static_cast<size_t>(top_k)
                       && cum < top_p) {
                    cum += probs[keep].first;
                    ++keep;
                }
                probs.resize(std::max<size_t>(1, keep));
            }
            std::vector<double> weights;
            weights.reserve(probs.size());
            for (const auto& p : probs) weights.push_back(p.first);
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            return probs[static_cast<size_t>(dist(rng))].second;
        };

        int token = draw_from_scores(scores, true);
        int rep = 0;
        const size_t begin = generated.size() > 10 ? generated.size() - 10 : 0;
        for (size_t i = begin; i < generated.size(); ++i) {
            if (generated[i] == token) ++rep;
        }
        if (rep >= 1 && token >= 0 && token < static_cast<int>(scores.size())) {
            scores[static_cast<size_t>(token)] = -std::numeric_limits<float>::infinity();
            token = draw_from_scores(scores, false);
        }
        return token;
    }

    std::vector<int64_t> run_llm(const std::vector<int64_t>& target_ids,
                                 int max_tokens,
                                 uint32_t seed,
                                 int* prefill_ms,
                                 int* ar_ms,
                                 bool* stopped_on_stop_token) {
        if (!has_conditioning) {
            throw std::runtime_error("CosyVoice3 conditioning is required");
        }
        if (stopped_on_stop_token) *stopped_on_stop_token = false;
        std::vector<int64_t> text_tokens_all = conditioning.prompt_text_ids;
        text_tokens_all.insert(text_tokens_all.end(), target_ids.begin(), target_ids.end());

        std::vector<int64_t> text_ids(kTextSlots, 0);
        std::vector<int64_t> prompt_ids(kPromptSpeechSlots, 0);
        const int text_len = std::min<int>(kTextSlots, text_tokens_all.size());
        const int prompt_len = std::min<int>(kPromptSpeechSlots,
            conditioning.llm_prompt_speech_tokens.size());
        std::copy_n(text_tokens_all.begin(), text_len, text_ids.begin());
        std::copy_n(conditioning.llm_prompt_speech_tokens.begin(), prompt_len,
                    prompt_ids.begin());
        int64_t text_len_v[1] = {text_len};
        int64_t prompt_len_v[1] = {prompt_len};
        const int target_text_len = std::max<int>(1, target_ids.size());
        const int min_tokens = std::max(1, target_text_len * 2);
        const int scaled_max_tokens =
            std::max(200, target_text_len * 10);
        const int max_decode = std::max(1, std::min(
            max_tokens,
            std::min(scaled_max_tokens,
                     kCacheSlots - text_len - prompt_len - 3)));

        const int64_t s_text[2] = {1, kTextSlots};
        const int64_t s_prompt[2] = {1, kPromptSpeechSlots};
        const int64_t s_len[1] = {1};
        OrtValue* in[4] = {
            make_i64(text_ids.data(), text_ids.size(), s_text, 2),
            make_i64(text_len_v, 1, s_len, 1),
            make_i64(prompt_ids.data(), prompt_ids.size(), s_prompt, 2),
            make_i64(prompt_len_v, 1, s_len, 1),
        };
        const char* in_names[4] = {
            "text_token_ids", "text_len",
            "prompt_speech_token_ids", "prompt_speech_len",
        };
        std::vector<const char*> out_names;
        out_names.push_back("logits");
        for (int i = 0; i < kLayers; ++i) {
            static thread_local std::vector<std::string> names;
            if (names.empty()) {
                names.reserve(2 * kLayers);
                char buf[64];
                for (int j = 0; j < kLayers; ++j) {
                    std::snprintf(buf, sizeof(buf), "present_key_%02d", j);
                    names.emplace_back(buf);
                }
                for (int j = 0; j < kLayers; ++j) {
                    std::snprintf(buf, sizeof(buf), "present_value_%02d", j);
                    names.emplace_back(buf);
                }
            }
            out_names.push_back(names[static_cast<size_t>(i)].c_str());
        }
        for (int i = 0; i < kLayers; ++i) {
            static thread_local std::vector<std::string> names;
            if (names.empty()) {
                names.reserve(2 * kLayers);
                char buf[64];
                for (int j = 0; j < kLayers; ++j) {
                    std::snprintf(buf, sizeof(buf), "present_key_%02d", j);
                    names.emplace_back(buf);
                }
                for (int j = 0; j < kLayers; ++j) {
                    std::snprintf(buf, sizeof(buf), "present_value_%02d", j);
                    names.emplace_back(buf);
                }
            }
            out_names.push_back(names[static_cast<size_t>(kLayers + i)].c_str());
        }
        std::vector<OrtValue*> outs(out_names.size(), nullptr);
        auto t0 = Clock::now();
        ort_check(api, api->Run(prefill, nullptr,
            in_names, in, 4,
            out_names.data(), out_names.size(), outs.data()));
        *prefill_ms = static_cast<int>(elapsed_ms(t0));
        for (auto* v : in) api->ReleaseValue(v);

        std::vector<float> logits = copy_f32(outs[0], 6761);
        std::vector<std::vector<float>> kv;
        kv.reserve(2 * kLayers);
        for (int i = 1; i < static_cast<int>(outs.size()); ++i) {
            kv.push_back(copy_f32(outs[static_cast<size_t>(i)], kKvFloats));
        }
        for (auto* v : outs) if (v) api->ReleaseValue(v);

        std::vector<int64_t> generated;
        generated.reserve(static_cast<size_t>(max_decode));
        std::mt19937 rng(seed);
        int64_t cache_position = text_len + prompt_len + 2;
        const int64_t s_tok[2] = {1, 1};
        const int64_t s_pos[1] = {1};
        const int64_t s_kv[4] = {1, 2, kCacheSlots, 64};
        const auto ar_t0 = Clock::now();
        for (int step_idx = 0; step_idx < max_decode; ++step_idx) {
            if (cancelled.load(std::memory_order_relaxed)) break;
            int tok = sample_token(logits, generated, min_tokens, rng, 25, 0.8);
            if (tok >= kSpeechTokenSize && tok < kSpeechTokenSize + 3) {
                if (stopped_on_stop_token) *stopped_on_stop_token = true;
                break;
            }
            if (tok >= kSpeechTokenSize) break;
            generated.push_back(tok);

            int64_t tok_v[1] = {tok};
            int64_t pos_v[1] = {cache_position};
            std::vector<const char*> step_inputs;
            std::vector<OrtValue*> step_vals;
            step_inputs.reserve(50);
            step_vals.reserve(50);
            step_inputs.push_back("speech_token_id");
            step_vals.push_back(make_i64(tok_v, 1, s_tok, 2));
            step_inputs.push_back("cache_position");
            step_vals.push_back(make_i64(pos_v, 1, s_pos, 1));
            std::vector<std::string> key_names;
            key_names.reserve(48);
            char buf[64];
            for (int i = 0; i < kLayers; ++i) {
                std::snprintf(buf, sizeof(buf), "past_key_%02d", i);
                key_names.emplace_back(buf);
                step_inputs.push_back(key_names.back().c_str());
                step_vals.push_back(make_f32(kv[static_cast<size_t>(i)].data(),
                                             kKvFloats, s_kv, 4));
            }
            for (int i = 0; i < kLayers; ++i) {
                std::snprintf(buf, sizeof(buf), "past_value_%02d", i);
                key_names.emplace_back(buf);
                step_inputs.push_back(key_names.back().c_str());
                step_vals.push_back(make_f32(kv[static_cast<size_t>(kLayers + i)].data(),
                                             kKvFloats, s_kv, 4));
            }

            std::vector<std::string> out_name_str;
            out_name_str.reserve(49);
            out_name_str.emplace_back("logits");
            for (int i = 0; i < kLayers; ++i) {
                std::snprintf(buf, sizeof(buf), "present_key_%02d", i);
                out_name_str.emplace_back(buf);
            }
            for (int i = 0; i < kLayers; ++i) {
                std::snprintf(buf, sizeof(buf), "present_value_%02d", i);
                out_name_str.emplace_back(buf);
            }
            std::vector<const char*> step_out_names;
            for (auto& s : out_name_str) step_out_names.push_back(s.c_str());
            std::vector<OrtValue*> step_outs(step_out_names.size(), nullptr);
            ort_check(api, api->Run(step, nullptr,
                step_inputs.data(), step_vals.data(), step_vals.size(),
                step_out_names.data(), step_out_names.size(), step_outs.data()));
            for (auto* v : step_vals) api->ReleaseValue(v);

            logits = copy_f32(step_outs[0], 6761);
            for (int i = 0; i < 2 * kLayers; ++i) {
                kv[static_cast<size_t>(i)] =
                    copy_f32(step_outs[static_cast<size_t>(1 + i)], kKvFloats);
            }
            for (auto* v : step_outs) if (v) api->ReleaseValue(v);
            ++cache_position;
        }
        *ar_ms = static_cast<int>(elapsed_ms(ar_t0));
        return generated;
    }

    std::vector<float> run_flow_hift(const std::vector<int64_t>& speech_tokens,
                                     int flow_steps,
                                     float cfg_rate,
                                     int* decode_ms,
                                     int* flow_frontend_ms,
                                     int* flow_estimator_ms,
                                     int* hift_ms) {
        if (!has_conditioning) {
            throw std::runtime_error("CosyVoice3 conditioning is required");
        }
        flow_steps = std::max(1, std::min(flow_steps, 32));
        if (!std::isfinite(cfg_rate) || cfg_rate < 0.0f) cfg_rate = 0.0f;
        std::vector<int64_t> merged = conditioning.flow_prompt_speech_tokens;
        merged.insert(merged.end(), speech_tokens.begin(), speech_tokens.end());
        if (merged.size() > kFlowTokenSlots) {
            throw std::runtime_error("CosyVoice3 flow token budget exceeded");
        }
        std::vector<int64_t> token_ids(kFlowTokenSlots, 0);
        for (size_t i = 0; i < merged.size(); ++i) {
            token_ids[i] = std::clamp<int64_t>(merged[i], 0, kSpeechTokenSize - 1);
        }
        int64_t token_len[1] = {static_cast<int64_t>(merged.size())};
        std::vector<float> prompt_feat(static_cast<size_t>(kMelSlots) * 80, 0.0f);
        const int prompt_frames = static_cast<int>(std::min<int64_t>(
            kMelSlots, conditioning.prompt_speech_feat_frames));
        const int src_frames = static_cast<int>(std::min<int64_t>(
            prompt_frames,
            static_cast<int64_t>(conditioning.prompt_speech_feat.size() / 80)));
        if (src_frames > 0) {
            std::memcpy(prompt_feat.data(), conditioning.prompt_speech_feat.data(),
                        static_cast<size_t>(src_frames) * 80 * sizeof(float));
        }
        int64_t prompt_feat_len[1] = {src_frames};
        if (conditioning.embedding.size() != 192) {
            throw std::runtime_error("CosyVoice3 conditioning embedding must have 192 floats");
        }

        const int64_t s_tok[2] = {1, kFlowTokenSlots};
        const int64_t s_len[1] = {1};
        const int64_t s_feat[3] = {1, kMelSlots, 80};
        const int64_t s_emb[2] = {1, 192};
        OrtValue* in[5] = {
            make_i64(token_ids.data(), token_ids.size(), s_tok, 2),
            make_i64(token_len, 1, s_len, 1),
            make_f32(prompt_feat.data(), prompt_feat.size(), s_feat, 3),
            make_i64(prompt_feat_len, 1, s_len, 1),
            make_f32(conditioning.embedding.data(), conditioning.embedding.size(), s_emb, 2),
        };
        const char* in_names[5] = {
            "token_ids", "token_len", "prompt_feat",
            "prompt_feat_len", "embedding",
        };
        const char* out_names[4] = {"mu", "mask", "spks", "cond"};
        OrtValue* outs[4] = {nullptr, nullptr, nullptr, nullptr};
        auto t0 = Clock::now();
        auto flow_frontend_t0 = Clock::now();
        ort_check(api, api->Run(flow_frontend, nullptr,
            in_names, in, 5, out_names, 4, outs));
        *flow_frontend_ms = static_cast<int>(elapsed_ms(flow_frontend_t0));
        for (auto* v : in) api->ReleaseValue(v);

        const int generated_frames =
            std::max(2, std::min(kMelSlots - src_frames,
                                 static_cast<int>(2 * speech_tokens.size())));
        if (generated_frames > kHiftFrames) {
            for (auto* v : outs) if (v) api->ReleaseValue(v);
            throw std::runtime_error("CosyVoice3 HiFT frame budget exceeded");
        }
        const int total_frames = src_frames + generated_frames;
        std::vector<float> mu_full = copy_f32(outs[0], static_cast<size_t>(80) * kMelSlots);
        std::vector<float> mask_full = copy_f32(outs[1], kMelSlots);
        std::vector<float> spks = copy_f32(outs[2], 80);
        std::vector<float> cond_full = copy_f32(outs[3], static_cast<size_t>(80) * kMelSlots);
        for (auto* v : outs) if (v) api->ReleaseValue(v);

        std::vector<float> mu(static_cast<size_t>(80) * total_frames);
        std::vector<float> cond(static_cast<size_t>(80) * total_frames);
        std::vector<float> mask(total_frames);
        for (int c = 0; c < 80; ++c) {
            std::copy_n(mu_full.data() + static_cast<size_t>(c) * kMelSlots,
                        total_frames, mu.data() + static_cast<size_t>(c) * total_frames);
            std::copy_n(cond_full.data() + static_cast<size_t>(c) * kMelSlots,
                        total_frames, cond.data() + static_cast<size_t>(c) * total_frames);
        }
        std::copy_n(mask_full.data(), total_frames, mask.data());

        std::mt19937 rng(1986);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> x(static_cast<size_t>(80) * total_frames);
        for (float& v : x) v = normal(rng);

        auto cosine_t = [flow_steps](int i) {
            constexpr float kPi = 3.14159265358979323846f;
            const float u = static_cast<float>(i) / static_cast<float>(flow_steps);
            return 1.0f - std::cos(u * kPi / 2.0f);
        };
        const int64_t s_dyn[3] = {2, 80, total_frames};
        const int64_t s_mask[3] = {2, 1, total_frames};
        const int64_t s_t[1] = {2};
        const int64_t s_spks[2] = {2, 80};
        auto flow_estimator_t0 = Clock::now();
        for (int i = 0; i < flow_steps; ++i) {
            std::vector<float> x2(static_cast<size_t>(2) * 80 * total_frames);
            std::vector<float> mu2(x2.size());
            std::vector<float> cond2(x2.size(), 0.0f);
            std::copy(x.begin(), x.end(), x2.begin());
            std::copy(x.begin(), x.end(), x2.begin() + x.size());
            std::copy(mu.begin(), mu.end(), mu2.begin());
            std::copy(mu.begin(), mu.end(), mu2.begin() + mu.size());
            std::copy(cond.begin(), cond.end(), cond2.begin());
            std::vector<float> mask2(static_cast<size_t>(2) * total_frames);
            std::copy(mask.begin(), mask.end(), mask2.begin());
            std::copy(mask.begin(), mask.end(), mask2.begin() + mask.size());
            std::vector<float> spks2(160);
            std::copy(spks.begin(), spks.end(), spks2.begin());
            std::copy(spks.begin(), spks.end(), spks2.begin() + 80);
            float t[2] = {cosine_t(i), cosine_t(i)};

            OrtValue* est_in[6] = {
                make_f32(x2.data(), x2.size(), s_dyn, 3),
                make_f32(mask2.data(), mask2.size(), s_mask, 3),
                make_f32(mu2.data(), mu2.size(), s_dyn, 3),
                make_f32(t, 2, s_t, 1),
                make_f32(spks2.data(), spks2.size(), s_spks, 2),
                make_f32(cond2.data(), cond2.size(), s_dyn, 3),
            };
            const char* est_in_names[6] = {"x", "mask", "mu", "t", "spks", "cond"};
            const char* est_out_names[1] = {"estimator_out"};
            OrtValue* est_out[1] = {nullptr};
            ort_check(api, api->Run(flow_estimator, nullptr,
                est_in_names, est_in, 6, est_out_names, 1, est_out));
            for (auto* v : est_in) api->ReleaseValue(v);
            std::vector<float> vel = copy_f32(est_out[0], x2.size());
            api->ReleaseValue(est_out[0]);
            const float dt = cosine_t(i + 1) - cosine_t(i);
            const float cond_scale = 1.0f + cfg_rate;
            for (size_t j = 0; j < x.size(); ++j) {
                const float guided = cond_scale * vel[j] - cfg_rate * vel[j + x.size()];
                x[j] += dt * guided;
            }
        }
        *flow_estimator_ms = static_cast<int>(elapsed_ms(flow_estimator_t0));

        OrtSession* hift_session = hift;
        int hift_frames = kHiftFrames;
        if (generated_frames <= kHiftFrames128 && hift_128) {
            hift_session = hift_128;
            hift_frames = kHiftFrames128;
        } else if (generated_frames <= kHiftFrames256 && hift_256) {
            hift_session = hift_256;
            hift_frames = kHiftFrames256;
        }

        std::vector<float> hift_in(static_cast<size_t>(80) * hift_frames, 0.0f);
        for (int c = 0; c < 80; ++c) {
            const float* src = x.data() + static_cast<size_t>(c) * total_frames + src_frames;
            float* dst = hift_in.data() + static_cast<size_t>(c) * hift_frames;
            std::copy_n(src, generated_frames, dst);
        }
        const int64_t s_hift[3] = {1, 80, hift_frames};
        OrtValue* hift_in_v = make_f32(hift_in.data(), hift_in.size(), s_hift, 3);
        const char* hift_in_names[1] = {"speech_feat"};
        const char* hift_out_names[1] = {"wav"};
        OrtValue* hift_out[1] = {nullptr};
        auto hift_t0 = Clock::now();
        ort_check(api, api->Run(hift_session, nullptr,
            hift_in_names, &hift_in_v, 1, hift_out_names, 1, hift_out));
        api->ReleaseValue(hift_in_v);
        std::vector<float> wav = copy_f32(hift_out[0], static_cast<size_t>(hift_frames) * 480);
        api->ReleaseValue(hift_out[0]);
        *hift_ms = static_cast<int>(elapsed_ms(hift_t0));
        wav.resize(static_cast<size_t>(generated_frames) * 480);
        *decode_ms = static_cast<int>(elapsed_ms(t0));
        return wav;
    }
};

OnnxCosyVoice3Tts::OnnxCosyVoice3Tts(const std::string& bundle_dir,
                                     bool hw_accel)
    : impl_(std::make_unique<Impl>(bundle_dir, hw_accel)) {}

OnnxCosyVoice3Tts::~OnnxCosyVoice3Tts() = default;

void OnnxCosyVoice3Tts::set_conditioning(Conditioning conditioning) {
    if (conditioning.prompt_speech_feat_frames <= 0) {
        conditioning.prompt_speech_feat_frames =
            static_cast<int64_t>(conditioning.prompt_speech_feat.size() / 80);
    }
    impl_->conditioning = std::move(conditioning);
    impl_->has_conditioning = true;
}

void OnnxCosyVoice3Tts::clear_conditioning() {
    impl_->conditioning = Conditioning{};
    impl_->has_conditioning = false;
}

bool OnnxCosyVoice3Tts::has_conditioning() const {
    return impl_->has_conditioning;
}

void OnnxCosyVoice3Tts::set_flow_steps(int steps) {
    flow_steps_ = std::max(1, std::min(steps, 32));
}

void OnnxCosyVoice3Tts::set_cfg_rate(float cfg_rate) {
    cfg_rate_ = std::isfinite(cfg_rate) ? std::max(0.0f, cfg_rate) : 0.0f;
}

void OnnxCosyVoice3Tts::cancel() {
    impl_->cancelled.store(true, std::memory_order_relaxed);
}

void OnnxCosyVoice3Tts::synthesize(const std::string& text,
                                   const std::string& /*language*/,
                                   TTSChunkCallback on_chunk) {
    if (!on_chunk) return;
    impl_->cancelled.store(false, std::memory_order_relaxed);
    seed_used_ = seed_ ? seed_ : std::random_device{}();
    tokens_generated_ = 0;
    stopped_on_stop_token_ = false;
    prefill_ms_ = -1;
    ar_ms_ = -1;
    audio_decode_ms_ = -1;
    flow_frontend_ms_ = -1;
    flow_estimator_ms_ = -1;
    hift_ms_ = -1;

    std::string prompt = instruction_.empty() ? text : instruction_ + " " + text;
    std::vector<int64_t> target = impl_->encode_text(prompt);
    int prefill_ms = 0;
    int ar_ms = 0;
    bool stopped_on_stop_token = false;
    auto speech_tokens = impl_->run_llm(target, max_steps_, seed_used_,
                                        &prefill_ms, &ar_ms,
                                        &stopped_on_stop_token);
    tokens_generated_ = static_cast<int>(speech_tokens.size());
    stopped_on_stop_token_ = stopped_on_stop_token;
    prefill_ms_ = prefill_ms;
    ar_ms_ = ar_ms;
    int decode_ms = 0;
    int flow_frontend_ms = 0;
    int flow_estimator_ms = 0;
    int hift_ms = 0;
    auto wav = impl_->run_flow_hift(speech_tokens, flow_steps_, cfg_rate_,
                                    &decode_ms,
                                    &flow_frontend_ms,
                                    &flow_estimator_ms,
                                    &hift_ms);
    audio_decode_ms_ = decode_ms;
    flow_frontend_ms_ = flow_frontend_ms;
    flow_estimator_ms_ = flow_estimator_ms;
    hift_ms_ = hift_ms;
    on_chunk(wav.data(), wav.size(), true);
}

std::string OnnxCosyVoice3Tts::helper_prompt_prefix() {
    return "You are a helpful assistant.<|endofprompt|>";
}

std::string OnnxCosyVoice3Tts::prompt_text_from_transcript(const std::string& transcript) {
    if (transcript.find("<|endofprompt|>") != std::string::npos) {
        return transcript;
    }
    return helper_prompt_prefix() + transcript;
}

std::vector<uint8_t> OnnxCosyVoice3Tts::encode_conditioning_blob(const Conditioning& c) {
    std::vector<uint8_t> out;
    const char magic[] = {'S','C','C','O','S','Y','3','\0'};
    out.insert(out.end(), magic, magic + sizeof(magic));
    write_u32(out, 1);
    write_u32(out, static_cast<uint32_t>(c.prompt_speech_feat_frames));
    write_vec(out, c.prompt_text_ids);
    write_vec(out, c.llm_prompt_speech_tokens);
    write_vec(out, c.flow_prompt_speech_tokens);
    write_vec(out, c.prompt_speech_feat);
    write_vec(out, c.embedding);
    return out;
}

OnnxCosyVoice3Tts::Conditioning
OnnxCosyVoice3Tts::decode_conditioning_blob(const uint8_t* data, size_t size) {
    if (!data || size < 12) throw std::runtime_error("CosyVoice3 conditioning blob empty");
    const uint8_t* p = data;
    const uint8_t* end = data + size;
    const char magic[] = {'S','C','C','O','S','Y','3','\0'};
    if (std::memcmp(p, magic, sizeof(magic)) != 0) {
        throw std::runtime_error("CosyVoice3 conditioning blob has bad magic");
    }
    p += sizeof(magic);
    const uint32_t version = read_u32(p, end);
    if (version != 1) throw std::runtime_error("CosyVoice3 conditioning blob version unsupported");
    Conditioning c;
    c.prompt_speech_feat_frames = read_u32(p, end);
    c.prompt_text_ids = read_vec<int64_t>(p, end);
    c.llm_prompt_speech_tokens = read_vec<int64_t>(p, end);
    c.flow_prompt_speech_tokens = read_vec<int64_t>(p, end);
    c.prompt_speech_feat = read_vec<float>(p, end);
    c.embedding = read_vec<float>(p, end);
    if (p != end) throw std::runtime_error("CosyVoice3 conditioning blob has trailing data");
    return c;
}

}  // namespace speech_core
