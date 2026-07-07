#include "speech_core/models/litert_indic_mio_tts.h"

#include "speech_core/audio/resampler.h"
#include "indic_mio_istft.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace speech_core {

namespace {

// Sample one id from full-vocab logits: repetition penalty over the generated
// ids, temperature, top-k, then top-p within the k candidates — the HF
// semantics the upstream reference (temperature 0.9, top_p 0.9) runs with.
int sample_logits(std::vector<float>& logits, const std::vector<int>& generated,
                  float temperature, int top_k, float top_p,
                  float repetition_penalty, std::mt19937& rng) {
    if (repetition_penalty != 1.0f) {
        for (int id : generated) {
            float& v = logits[static_cast<size_t>(id)];
            v = v > 0 ? v / repetition_penalty : v * repetition_penalty;
        }
    }
    if (temperature <= 0.0f) {  // greedy
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) -
                                logits.begin());
    }

    const int k = std::clamp(top_k, 1, static_cast<int>(logits.size()));
    std::vector<int> idx(logits.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[static_cast<size_t>(a)] >
                                                 logits[static_cast<size_t>(b)]; });

    // Softmax over the k candidates (temperature applied), then nucleus cut.
    std::vector<double> probs(static_cast<size_t>(k));
    const double maxv = logits[static_cast<size_t>(idx[0])];
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        const double p =
            std::exp((logits[static_cast<size_t>(idx[static_cast<size_t>(i)])] - maxv) /
                     temperature);
        probs[static_cast<size_t>(i)] = p;
        sum += p;
    }
    int keep = k;
    if (top_p < 1.0f) {
        double acc = 0.0;
        for (int i = 0; i < k; ++i) {
            acc += probs[static_cast<size_t>(i)] / sum;
            if (acc >= top_p) { keep = i + 1; break; }
        }
    }
    double keep_sum = 0.0;
    for (int i = 0; i < keep; ++i) keep_sum += probs[static_cast<size_t>(i)];
    std::uniform_real_distribution<double> uni(0.0, keep_sum);
    double r = uni(rng);
    for (int i = 0; i < keep; ++i) {
        r -= probs[static_cast<size_t>(i)];
        if (r <= 0.0) return idx[static_cast<size_t>(i)];
    }
    return idx[static_cast<size_t>(keep - 1)];
}

}  // namespace

LiteRTIndicMioTts::LiteRTIndicMioTts(const std::string& text_prefill_path,
                                     const std::string& token_step_path,
                                     const std::string& audio_decoder_path,
                                     const std::string& ref_encoder_path,
                                     const std::string& tokenizer_path,
                                     bool hw_accel) {
    text_prefill_path_     = text_prefill_path;
    text_prefill_hw_accel_ = hw_accel;

    auto& engine = LiteRTEngine::get();
    engine.load(token_step_path, hw_accel, &token_step_model_, &token_step_compiled_);
    engine.load(audio_decoder_path, hw_accel, &decoder_model_, &decoder_compiled_);
    engine.load(ref_encoder_path, hw_accel, &ref_encoder_model_, &ref_encoder_compiled_);

    tokenizer_ = std::make_unique<IndicMioTokenizer>(tokenizer_path);
    istft_     = std::make_unique<IndicMioIstft>(kNfft, kHop);
}

LiteRTIndicMioTts::~LiteRTIndicMioTts() {
    if (ref_encoder_compiled_) LiteRtDestroyCompiledModel(ref_encoder_compiled_);
    if (ref_encoder_model_)    LiteRtDestroyModel(ref_encoder_model_);
    if (decoder_compiled_)     LiteRtDestroyCompiledModel(decoder_compiled_);
    if (decoder_model_)        LiteRtDestroyModel(decoder_model_);
    if (token_step_compiled_)  LiteRtDestroyCompiledModel(token_step_compiled_);
    if (token_step_model_)     LiteRtDestroyModel(token_step_model_);
    release_text_prefill();
}

void LiteRTIndicMioTts::load_text_prefill() {
    if (text_prefill_compiled_) return;
    LiteRTEngine::get().load(text_prefill_path_, text_prefill_hw_accel_,
                             &text_prefill_model_, &text_prefill_compiled_);
}

void LiteRTIndicMioTts::release_text_prefill() {
    if (text_prefill_compiled_) {
        LiteRtDestroyCompiledModel(text_prefill_compiled_);
        text_prefill_compiled_ = nullptr;
    }
    if (text_prefill_model_) {
        LiteRtDestroyModel(text_prefill_model_);
        text_prefill_model_ = nullptr;
    }
    LiteRTEngine::get().release_buffer(text_prefill_path_);
}

void LiteRTIndicMioTts::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
}

void LiteRTIndicMioTts::set_max_new_tokens(int n) {
    max_new_tokens_ = std::clamp(n, 1, std::min(kMaxSeq - kPromptLen, kDecoderTokens));
}

// ---------------------------------------------------------------------------
// Reference conditioning — encode once, cache the 128-float embedding.
// ---------------------------------------------------------------------------

void LiteRTIndicMioTts::clear_reference() {
    global_embedding_.clear();
    global_embedding_.shrink_to_fit();
}

void LiteRTIndicMioTts::set_reference(const float* pcm, size_t length,
                                      int sample_rate) {
    clear_reference();
    if (!pcm || length == 0) return;

    std::vector<float> mono;
    if (sample_rate != output_sample_rate()) {
        mono = Resampler::resample(pcm, length, sample_rate, output_sample_rate());
    } else {
        mono.assign(pcm, pcm + length);
    }
    if (mono.empty()) return;

    // Crop, don't pad, whenever possible: the encoder pools over all frames
    // and zero padding measurably dilutes the embedding (bundle model card).
    std::vector<float> input(kRefSamples, 0.0f);
    if (mono.size() >= static_cast<size_t>(kRefSamples)) {
        const size_t start = (mono.size() - kRefSamples) / 2;
        std::copy(mono.begin() + static_cast<std::ptrdiff_t>(start),
                  mono.begin() + static_cast<std::ptrdiff_t>(start + kRefSamples),
                  input.begin());
    } else {
        const size_t pad = (static_cast<size_t>(kRefSamples) - mono.size()) / 2;
        std::copy(mono.begin(), mono.end(),
                  input.begin() + static_cast<std::ptrdiff_t>(pad));
    }

    auto env = LiteRTEngine::get().env();
    auto t_in  = make_type(kLiteRtElementTypeFloat32, {1, kRefSamples});
    auto t_out = make_type(kLiteRtElementTypeFloat32, {1, kGlobalDim});
    LiteRtHostBuffer in (env, t_in,  input.size() * sizeof(float), input.data());
    LiteRtHostBuffer out(env, t_out, kGlobalDim * sizeof(float));
    LiteRtTensorBuffer ins[1]  = { in.raw() };
    LiteRtTensorBuffer outs[1] = { out.raw() };
    litert_check(LiteRtRunCompiledModel(ref_encoder_compiled_, 0, 1, ins, 1, outs),
                 "indic-mio ref_encoder Run");
    global_embedding_.resize(kGlobalDim);
    out.read(global_embedding_.data(), kGlobalDim * sizeof(float));
}

// ---------------------------------------------------------------------------
// Prompt — the bundle's chat template with control ids inserted by id (the
// exact construction pinned by test_indic_mio_tokenizer's prompt fixture).
// ---------------------------------------------------------------------------

std::vector<int64_t> LiteRTIndicMioTts::build_prompt(const std::string& text,
                                                     int& real_len) const {
    std::vector<int> ids;
    ids.push_back(kImStartId);
    for (int v : tokenizer_->encode("user\n" + text)) ids.push_back(v);
    ids.push_back(kImEndId);
    for (int v : tokenizer_->encode("\n")) ids.push_back(v);
    ids.push_back(kImStartId);
    for (int v : tokenizer_->encode("assistant\n")) ids.push_back(v);

    if (static_cast<int>(ids.size()) > kPromptLen) {
        throw std::runtime_error(
            "Indic-Mio: prompt is " + std::to_string(ids.size()) +
            " tokens; the bucket is " + std::to_string(kPromptLen) +
            " — split the text upstream");
    }
    real_len = static_cast<int>(ids.size());
    std::vector<int64_t> padded(kPromptLen, kEndOfTextId);
    for (size_t i = 0; i < ids.size(); ++i) padded[i] = ids[i];
    return padded;
}

// ---------------------------------------------------------------------------
// synthesize — prefill → AR loop → masked decode → host ISTFT. Buffered: one
// final chunk (whole-utterance synthesis; the AR loop dominates latency).
// ---------------------------------------------------------------------------

void LiteRTIndicMioTts::synthesize(const std::string& text,
                                   const std::string& /*language*/,
                                   TTSChunkCallback on_chunk) {
    if (!on_chunk) return;
    cancelled_.store(false, std::memory_order_relaxed);
    tokens_generated_ = 0;
    stopped_on_eos_   = false;

    int prompt_len = 0;
    const std::vector<int64_t> prompt = build_prompt(text, prompt_len);

    uint32_t rng_seed = seed_;
    if (rng_seed == 0) {
        std::random_device rd;
        rng_seed = rd();
    }
    seed_used_ = rng_seed;
    std::mt19937 rng(rng_seed);

    auto env = LiteRTEngine::get().env();
    // One K (or V) tensor: [layers, 1, kv_heads, max_seq, head_dim].
    const size_t kv_floats =
        static_cast<size_t>(kLayers) * kKvHeads * kMaxSeq * kHeadDim;
    std::vector<float> logits(static_cast<size_t>(kVocab));

    auto t_ids    = make_type(kLiteRtElementTypeInt64,   {1, kPromptLen});
    auto t_scalar = make_type(kLiteRtElementTypeInt64,   {});
    auto t_logits = make_type(kLiteRtElementTypeFloat32, {1, kVocab});
    auto t_kv     = make_type(kLiteRtElementTypeFloat32,
                              {kLayers, 1, kKvHeads, kMaxSeq, kHeadDim});
    auto t_id1    = make_type(kLiteRtElementTypeInt64,   {1, 1});

    // KV ping-pong: the step's outputs become the next step's inputs without
    // a 59 MB×2 host copy per token (the VoxCPM2 hoisted-buffer lesson, taken
    // one step further).
    LiteRtHostBuffer kv_a_k(env, t_kv, kv_floats * sizeof(float));
    LiteRtHostBuffer kv_a_v(env, t_kv, kv_floats * sizeof(float));
    LiteRtHostBuffer kv_b_k(env, t_kv, kv_floats * sizeof(float));
    LiteRtHostBuffer kv_b_v(env, t_kv, kv_floats * sizeof(float));
    LiteRtHostBuffer* kv_in_k  = &kv_a_k;
    LiteRtHostBuffer* kv_in_v  = &kv_a_v;
    LiteRtHostBuffer* kv_out_k = &kv_b_k;
    LiteRtHostBuffer* kv_out_v = &kv_b_v;

    // --- 1. Prefill (lazy-load the 1.1 GB graph, release right after).
    {
        std::lock_guard<std::mutex> lock(prefill_mutex_);
        load_text_prefill();
        const int64_t last_index = prompt_len - 1;
        LiteRtHostBuffer in_ids (env, t_ids, prompt.size() * sizeof(int64_t), prompt.data());
        LiteRtHostBuffer in_last(env, t_scalar, sizeof(int64_t), &last_index);
        LiteRtHostBuffer out_logits(env, t_logits, logits.size() * sizeof(float));
        LiteRtTensorBuffer ins[2]  = { in_ids.raw(), in_last.raw() };
        LiteRtTensorBuffer outs[3] = { out_logits.raw(), kv_in_k->raw(), kv_in_v->raw() };
        litert_check(LiteRtRunCompiledModel(text_prefill_compiled_, 0, 2, ins, 3, outs),
                     "indic-mio text_prefill Run");
        out_logits.read(logits.data(), logits.size() * sizeof(float));
        release_text_prefill();
    }

    // --- 2. AR loop.
    std::vector<int> generated;
    std::vector<int> codes;
    generated.reserve(static_cast<size_t>(max_new_tokens_));
    codes.reserve(static_cast<size_t>(max_new_tokens_));

    LiteRtHostBuffer in_id (env, t_id1,    sizeof(int64_t));
    LiteRtHostBuffer in_pos(env, t_id1,    sizeof(int64_t));
    LiteRtHostBuffer in_wi (env, t_scalar, sizeof(int64_t));
    LiteRtHostBuffer out_logits(env, t_logits, logits.size() * sizeof(float));

    for (int step = 0; step < max_new_tokens_; ++step) {
        if (cancelled_.load(std::memory_order_relaxed)) break;

        // EOS is suppressed until the model has produced at least one speech
        // token, so a take can never be empty.
        if (codes.empty()) {
            logits[static_cast<size_t>(kImEndId)]     = -1e30f;
            logits[static_cast<size_t>(kEndOfTextId)] = -1e30f;
        }
        const int next = sample_logits(logits, generated, temperature_, top_k_,
                                       top_p_, repetition_penalty_, rng);
        if (next == kImEndId || next == kEndOfTextId) {
            stopped_on_eos_ = true;
            break;
        }
        generated.push_back(next);
        if (next >= kSpeechOffset && next < kSpeechOffset + kSpeechCount) {
            codes.push_back(next - kSpeechOffset);
            if (static_cast<int>(codes.size()) >= max_new_tokens_) break;
        }

        const int64_t position = prompt_len + step;
        const int64_t id64     = next;
        in_id .write(&id64,     sizeof(int64_t));
        in_pos.write(&position, sizeof(int64_t));
        in_wi .write(&position, sizeof(int64_t));

        LiteRtTensorBuffer ins[5]  = { in_id.raw(), in_pos.raw(), in_wi.raw(),
                                       kv_in_k->raw(), kv_in_v->raw() };
        LiteRtTensorBuffer outs[3] = { out_logits.raw(), kv_out_k->raw(), kv_out_v->raw() };
        litert_check(LiteRtRunCompiledModel(token_step_compiled_, 0, 5, ins, 3, outs),
                     "indic-mio token_step Run");
        out_logits.read(logits.data(), logits.size() * sizeof(float));
        std::swap(kv_in_k, kv_out_k);
        std::swap(kv_in_v, kv_out_v);

        // Guard the KV budget: position for the NEXT write must stay < kMaxSeq.
        if (prompt_len + step + 1 >= kMaxSeq - 1) break;
    }
    tokens_generated_ = static_cast<int>(codes.size());

    if (codes.empty()) {
        on_chunk(nullptr, 0, /*is_final=*/true);
        return;
    }

    // --- 3. Masked decode + host ISTFT.
    const int n = static_cast<int>(codes.size());
    std::vector<int64_t> padded_codes(kDecoderTokens, 0);
    for (int i = 0; i < n; ++i) padded_codes[static_cast<size_t>(i)] = codes[static_cast<size_t>(i)];
    std::vector<float> global(kGlobalDim, 0.0f);
    if (!global_embedding_.empty()) {
        std::copy(global_embedding_.begin(), global_embedding_.end(), global.begin());
    }

    const int frames = kDecoderTokens * 2;  // 768 STFT frames for the bucket
    const int bins   = kNfft / 2 + 1;
    std::vector<float> spec_real(static_cast<size_t>(bins) * frames);
    std::vector<float> spec_imag(static_cast<size_t>(bins) * frames);
    {
        auto t_codes = make_type(kLiteRtElementTypeInt64,   {1, kDecoderTokens});
        auto t_glob  = make_type(kLiteRtElementTypeFloat32, {1, kGlobalDim});
        auto t_spec  = make_type(kLiteRtElementTypeFloat32, {1, bins, frames});
        const int64_t valid = n;
        LiteRtHostBuffer in_codes(env, t_codes, padded_codes.size() * sizeof(int64_t),
                                  padded_codes.data());
        LiteRtHostBuffer in_glob (env, t_glob, global.size() * sizeof(float), global.data());
        LiteRtHostBuffer in_valid(env, t_scalar, sizeof(int64_t), &valid);
        LiteRtHostBuffer out_real(env, t_spec, spec_real.size() * sizeof(float));
        LiteRtHostBuffer out_imag(env, t_spec, spec_imag.size() * sizeof(float));
        LiteRtTensorBuffer ins[3]  = { in_codes.raw(), in_glob.raw(), in_valid.raw() };
        LiteRtTensorBuffer outs[2] = { out_real.raw(), out_imag.raw() };
        litert_check(LiteRtRunCompiledModel(decoder_compiled_, 0, 3, ins, 2, outs),
                     "indic-mio audio_decoder Run");
        out_real.read(spec_real.data(), spec_real.size() * sizeof(float));
        out_imag.read(spec_imag.data(), spec_imag.size() * sizeof(float));
    }

    std::vector<float> pcm = istft_->synthesize(spec_real.data(), spec_imag.data(), frames);
    const size_t keep = std::min(pcm.size(), static_cast<size_t>(n) * kSamplesPerToken);
    on_chunk(pcm.data(), keep, /*is_final=*/true);
}

}  // namespace speech_core
