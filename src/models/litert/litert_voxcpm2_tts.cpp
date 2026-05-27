#include "speech_core/models/litert_voxcpm2_tts.h"

#include "speech_core/models/litert_engine.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>

namespace speech_core {

namespace {

// Set the i-th input tensor on `interp` from a host buffer. The buffer is
// expected to be the exact size of the tensor in bytes.
void set_input(TfLiteInterpreter* interp, int idx, const void* data, size_t bytes,
               const char* what) {
    TfLiteTensor* t = TfLiteInterpreterGetInputTensor(interp, idx);
    const size_t actual = TfLiteTensorByteSize(t);
    if (actual != bytes) {
        LOGE("%s: byte mismatch — tensor expects %zu, we passed %zu (rank=%d type=%d)",
             what, actual, bytes, TfLiteTensorNumDims(t),
             static_cast<int>(TfLiteTensorType(t)));
    }
    litert_check(TfLiteTensorCopyFromBuffer(t, data, bytes), what);
}

// Read the i-th output tensor from `interp` into a host buffer of `bytes` size.
void get_output(const TfLiteInterpreter* interp, int idx, void* data, size_t bytes,
                const char* what) {
    const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interp, idx);
    litert_check(TfLiteTensorCopyToBuffer(t, data, bytes), what);
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

LiteRTVoxCPM2Tts::LiteRTVoxCPM2Tts(const std::string& text_prefill_path,
                                    const std::string& token_step_path,
                                    const std::string& audio_encoder_path,
                                    const std::string& audio_decoder_path,
                                    const std::string& tokenizer_path,
                                    bool hw_accel)
{
    auto& engine = LiteRTEngine::get();
    // EXPERIMENT: load in REVERSE order. The Python reference frees each
    // interpreter before constructing the next; we hold all 4 alive at once.
    // If TFLite's op resolver / delegate state has any inter-interpreter
    // contamination, the LAST one loaded should be the only "clean" one.
    audio_decoder_ = engine.load(audio_decoder_path, hw_accel, &audio_decoder_model_);
    audio_encoder_ = engine.load(audio_encoder_path, hw_accel, &audio_encoder_model_);
    token_step_    = engine.load(token_step_path,    hw_accel, &token_step_model_);
    text_prefill_  = engine.load(text_prefill_path,  hw_accel, &text_prefill_model_);

    // Diagnostic — log input/output counts and per-input byte sizes for
    // text_prefill so we can compare against the Python reference (which
    // works on the same .tflite). If a tensor reports byte_size 0 here,
    // AllocateTensors didn't fully initialise it, which is the most
    // likely cause of "Input tensor N lacks data" at first Invoke.
    {
        const int nin  = TfLiteInterpreterGetInputTensorCount(text_prefill_);
        const int nout = TfLiteInterpreterGetOutputTensorCount(text_prefill_);
        LOGI("text_prefill inputs=%d outputs=%d", nin, nout);
        for (int i = 0; i < nin; ++i) {
            const TfLiteTensor* t = TfLiteInterpreterGetInputTensor(text_prefill_, i);
            LOGI("  input[%d] name=%s rank=%d type=%d bytes=%zu",
                 i, TfLiteTensorName(t) ? TfLiteTensorName(t) : "?",
                 TfLiteTensorNumDims(t), static_cast<int>(TfLiteTensorType(t)),
                 TfLiteTensorByteSize(t));
        }
    }

    tokenizer_         = std::make_unique<VoxCPM2Tokenizer>(tokenizer_path);
    audio_start_token_ = tokenizer_->token_id("<|audio_start|>");
    if (audio_start_token_ < 0) {
        throw std::runtime_error(
            "LiteRT VoxCPM2: tokenizer is missing <|audio_start|> — bundle is malformed");
    }
}

LiteRTVoxCPM2Tts::~LiteRTVoxCPM2Tts() {
    if (audio_decoder_)       TfLiteInterpreterDelete(audio_decoder_);
    if (audio_decoder_model_) TfLiteModelDelete(audio_decoder_model_);
    if (audio_encoder_)       TfLiteInterpreterDelete(audio_encoder_);
    if (audio_encoder_model_) TfLiteModelDelete(audio_encoder_model_);
    if (token_step_)          TfLiteInterpreterDelete(token_step_);
    if (token_step_model_)    TfLiteModelDelete(token_step_model_);
    if (text_prefill_)        TfLiteInterpreterDelete(text_prefill_);
    if (text_prefill_model_)  TfLiteModelDelete(text_prefill_model_);
}

void LiteRTVoxCPM2Tts::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// synthesize() — port of the reference Python smoke loop. Mirrors the order
// of operations in speech-models/models/voxcpm2/export/smoke_litert_roundtrip.py
// step-for-step so cross-checking against that script is straightforward.
// ---------------------------------------------------------------------------

void LiteRTVoxCPM2Tts::synthesize(const std::string& text,
                                   const std::string& /*language*/,
                                   TTSChunkCallback on_chunk)
{
    if (!on_chunk) return;
    cancelled_.store(false, std::memory_order_relaxed);

    // --- 1. Tokenize: VoxCPM2 was trained on "({instruction}){text}" prompts.
    // The audio_start token signals the model to begin generating audio.
    std::string prompt = "(" + instruction_ + ")" + text;
    std::vector<int> ids = tokenizer_->encode(prompt);
    ids.push_back(audio_start_token_);
    if (ids.size() > static_cast<size_t>(kMaxText)) ids.resize(kMaxText);
    const int context_length = static_cast<int>(ids.size());

    std::vector<int64_t> text_tokens(kMaxText, 0);
    std::vector<float>   text_mask  (kMaxText, 0.0f);
    for (int i = 0; i < context_length; ++i) {
        text_tokens[i] = ids[i];
        text_mask[i]   = 1.0f;
    }
    // No reference audio — feed zero conditioning. Voice cloning would replace
    // these with the audio_encoder output (40 patches × [4, 64]) padded out.
    std::vector<float> audio_feats(kMaxText * kPredFeatFloats, 0.0f);
    std::vector<float> audio_mask (kMaxText, 0.0f);
    const int64_t context_length_scalar = context_length;

    // --- 2. text_prefill — populate initial hiddens + caches.
    set_input(text_prefill_, 0, text_tokens.data(),
              text_tokens.size() * sizeof(int64_t), "prefill text_tokens");
    set_input(text_prefill_, 1, text_mask.data(),
              text_mask.size() * sizeof(float),    "prefill text_mask");
    set_input(text_prefill_, 2, audio_feats.data(),
              audio_feats.size() * sizeof(float),  "prefill audio_feats");
    set_input(text_prefill_, 3, audio_mask.data(),
              audio_mask.size() * sizeof(float),   "prefill audio_mask");
    set_input(text_prefill_, 4, &context_length_scalar,
              sizeof(int64_t),                     "prefill context_length");
    litert_check(TfLiteInterpreterInvoke(text_prefill_), "prefill Invoke");

    std::vector<float> lm_hidden       (kHidden);
    std::vector<float> residual_hidden (kHidden);
    std::vector<float> prefix_feat_cond(kPredFeatFloats);
    std::vector<float> base_prefill    (kBasePrefillFloats);
    std::vector<float> residual_prefill(kResidualPrefillFloats);
    get_output(text_prefill_, 0, lm_hidden.data(),        lm_hidden.size() * 4,        "prefill lm_hidden");
    get_output(text_prefill_, 1, residual_hidden.data(),  residual_hidden.size() * 4,  "prefill residual_hidden");
    get_output(text_prefill_, 2, prefix_feat_cond.data(), prefix_feat_cond.size() * 4, "prefill prefix_feat_cond");
    get_output(text_prefill_, 3, base_prefill.data(),     base_prefill.size() * 4,     "prefill base_cache");
    get_output(text_prefill_, 4, residual_prefill.data(), residual_prefill.size() * 4, "prefill residual_cache");

    // --- 3. Grow caches 512 → 2560 by zero-padding axis 4. The cache layout
    // is [2 (K/V), layers, 1 (batch), kv_heads, seq, head_dim]; we copy the
    // first 512 seq slots and leave the remaining 2048 zeroed.
    auto pad_cache = [](const std::vector<float>& src, std::vector<float>& dst,
                        int layers, int kv_heads, int head_dim) {
        constexpr int kPrefSeq = 512;
        constexpr int kFullSeq = 2560;
        std::fill(dst.begin(), dst.end(), 0.0f);
        const size_t per_kv      = static_cast<size_t>(layers) * kv_heads;
        const size_t src_row     = static_cast<size_t>(kPrefSeq) * head_dim;
        const size_t dst_row     = static_cast<size_t>(kFullSeq) * head_dim;
        for (int kv = 0; kv < 2; ++kv) {
            for (size_t i = 0; i < per_kv; ++i) {
                const float* src_ptr = src.data() + (kv * per_kv + i) * src_row;
                float*       dst_ptr = dst.data() + (kv * per_kv + i) * dst_row;
                std::memcpy(dst_ptr, src_ptr, src_row * sizeof(float));
            }
        }
    };
    std::vector<float> base_cache    (kBaseCacheFloats);
    std::vector<float> residual_cache(kResidualCacheFloats);
    pad_cache(base_prefill,     base_cache,     28, 2, 128);
    pad_cache(residual_prefill, residual_cache,  8, 2, 128);
    base_prefill    .clear(); base_prefill    .shrink_to_fit();
    residual_prefill.clear(); residual_prefill.shrink_to_fit();

    // --- 4. AR loop. Each token_step consumes the freshest hiddens + caches
    // and emits one 256-float pred_feat. Every 64 features are concatenated
    // into the decoder's fixed [1, 64, 256] latent slot and streamed out.
    std::vector<float>   pred_feat(kPredFeatFloats);
    std::vector<float>   stop_logits(2);
    std::vector<float>   next_lm(kHidden);
    std::vector<float>   next_residual(kHidden);
    std::vector<float>   noise(kPredFeatFloats);
    std::vector<float>   feature_buffer; feature_buffer.reserve(kDecoderInputFloats);
    std::vector<float>   decoder_input(kDecoderInputFloats, 0.0f);
    std::vector<float>   pcm(kDecoderOutputFloats);
    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    auto flush_decoder = [&](size_t valid_steps, bool is_final) {
        std::fill(decoder_input.begin(), decoder_input.end(), 0.0f);
        // feature_buffer is [N, 4, 64] in flattened pred_feat order (one
        // pred_feat appended per step). The decoder wants [1, 64, 256] where
        // axis 1 is the frame index and axis 2 is the flattened [4, 64].
        // The Python reference does: concat(axis=1).transpose(0, 2, 1), which
        // for our linear buffer is: pred_feat[step, q, f] -> decoder[step, q*64+f]
        // — i.e. each pred_feat's 256 floats slot directly into one decoder
        // frame. So a plain memcpy of the first `valid_steps * 256` floats
        // suffices, leaving the remainder zero-padded out to 64 frames.
        const size_t copy_floats = std::min<size_t>(
            valid_steps * kPredFeatFloats, kDecoderInputFloats);
        std::memcpy(decoder_input.data(), feature_buffer.data(),
                    copy_floats * sizeof(float));

        set_input(audio_decoder_, 0, decoder_input.data(),
                  decoder_input.size() * sizeof(float), "decoder latent");
        litert_check(TfLiteInterpreterInvoke(audio_decoder_), "decoder Invoke");
        get_output(audio_decoder_, 0, pcm.data(), pcm.size() * sizeof(float), "decoder pcm");

        // Trim the decoder window to the audio actually generated by the LM.
        const size_t valid_samples = std::min<size_t>(
            static_cast<size_t>(valid_steps) * kSamplesPerStep, pcm.size());
        on_chunk(pcm.data(), valid_samples, is_final);

        feature_buffer.clear();
    };

    int  steps_done       = 0;
    int  steps_in_chunk   = 0;
    bool stopped_by_model = false;

    for (int step = 0; step < max_steps_; ++step) {
        if (cancelled_.load(std::memory_order_relaxed)) break;

        for (float& v : noise) v = normal(rng);
        const int64_t position_id = static_cast<int64_t>(context_length + step);

        set_input(token_step_, 0, lm_hidden.data(),        lm_hidden.size() * 4,        "step lm_hidden");
        set_input(token_step_, 1, residual_hidden.data(),  residual_hidden.size() * 4,  "step residual_hidden");
        set_input(token_step_, 2, prefix_feat_cond.data(), prefix_feat_cond.size() * 4, "step prefix_feat_cond");
        set_input(token_step_, 3, base_cache.data(),       base_cache.size() * 4,       "step base_cache");
        set_input(token_step_, 4, residual_cache.data(),   residual_cache.size() * 4,   "step residual_cache");
        set_input(token_step_, 5, &position_id,            sizeof(int64_t),             "step position_id");
        set_input(token_step_, 6, noise.data(),            noise.size() * 4,            "step noise");
        litert_check(TfLiteInterpreterInvoke(token_step_), "step Invoke");

        get_output(token_step_, 0, pred_feat.data(),     pred_feat.size() * 4,     "step pred_feat");
        get_output(token_step_, 1, stop_logits.data(),   stop_logits.size() * 4,   "step stop_logits");
        get_output(token_step_, 2, next_lm.data(),       next_lm.size() * 4,       "step next_lm_hidden");
        get_output(token_step_, 3, next_residual.data(), next_residual.size() * 4, "step next_residual_hidden");
        get_output(token_step_, 4, base_cache.data(),    base_cache.size() * 4,    "step base_cache_out");
        get_output(token_step_, 5, residual_cache.data(), residual_cache.size() * 4, "step residual_cache_out");

        // Roll forward — the next step consumes the freshest hiddens, and
        // prefix_feat_cond is *replaced* by the latest pred_feat. The graph
        // re-uses that slot as the AR conditioning signal.
        lm_hidden        = next_lm;
        residual_hidden  = next_residual;
        prefix_feat_cond = pred_feat;

        feature_buffer.insert(feature_buffer.end(), pred_feat.begin(), pred_feat.end());
        ++steps_done;
        ++steps_in_chunk;

        const bool stop_signal = stop_logits[1] > stop_logits[0]
                              && steps_done > min_stop_steps_;
        if (stop_signal) { stopped_by_model = true; }

        if (steps_in_chunk == kFramesPerChunk) {
            flush_decoder(kFramesPerChunk, /*is_final=*/false);
            steps_in_chunk = 0;
        }

        if (stop_signal) break;
    }

    // Drain any partial chunk; emit the final-marker callback even if the
    // last frame landed exactly on a 64-step boundary so consumers always get
    // a tail signal.
    if (steps_in_chunk > 0) {
        flush_decoder(static_cast<size_t>(steps_in_chunk), /*is_final=*/true);
    } else {
        on_chunk(nullptr, 0, /*is_final=*/true);
    }

    (void)stopped_by_model;  // currently informational only
}

}  // namespace speech_core
