#include "speech_core/models/litert_voxcpm2_tts.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>

namespace speech_core {

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

LiteRTVoxCPM2Tts::LiteRTVoxCPM2Tts(const std::string& text_prefill_path,
                                    const std::string& token_step_path,
                                    const std::string& audio_encoder_path,
                                    const std::string& audio_decoder_path,
                                    const std::string& tokenizer_path,
                                    bool hw_accel)
{
    auto& engine = LiteRTEngine::get();
    engine.load(text_prefill_path,  hw_accel, &text_prefill_model_,  &text_prefill_compiled_);
    engine.load(token_step_path,    hw_accel, &token_step_model_,    &token_step_compiled_);
    engine.load(audio_encoder_path, hw_accel, &audio_encoder_model_, &audio_encoder_compiled_);
    engine.load(audio_decoder_path, hw_accel, &audio_decoder_model_, &audio_decoder_compiled_);

    tokenizer_         = std::make_unique<VoxCPM2Tokenizer>(tokenizer_path);
    audio_start_token_ = tokenizer_->token_id("<|audio_start|>");
    if (audio_start_token_ < 0) {
        throw std::runtime_error(
            "LiteRT VoxCPM2: tokenizer is missing <|audio_start|> — bundle is malformed");
    }
}

LiteRTVoxCPM2Tts::~LiteRTVoxCPM2Tts() {
    if (audio_decoder_compiled_) LiteRtDestroyCompiledModel(audio_decoder_compiled_);
    if (audio_decoder_model_)    LiteRtDestroyModel(audio_decoder_model_);
    if (audio_encoder_compiled_) LiteRtDestroyCompiledModel(audio_encoder_compiled_);
    if (audio_encoder_model_)    LiteRtDestroyModel(audio_encoder_model_);
    if (token_step_compiled_)    LiteRtDestroyCompiledModel(token_step_compiled_);
    if (token_step_model_)       LiteRtDestroyModel(token_step_model_);
    if (text_prefill_compiled_)  LiteRtDestroyCompiledModel(text_prefill_compiled_);
    if (text_prefill_model_)     LiteRtDestroyModel(text_prefill_model_);
}

void LiteRTVoxCPM2Tts::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// synthesize() — mirrors speech-models/.../smoke_litert_roundtrip.py
// ---------------------------------------------------------------------------

void LiteRTVoxCPM2Tts::synthesize(const std::string& text,
                                   const std::string& /*language*/,
                                   TTSChunkCallback on_chunk)
{
    if (!on_chunk) return;
    cancelled_.store(false, std::memory_order_relaxed);

    // --- 1. Tokenize: VoxCPM2 was trained on "({instruction}){text}" prompts.
    // <|audio_start|> signals the model to begin generating audio.
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
    // No reference audio — feed zero audio_feats/audio_mask. Voice cloning
    // would replace these with the audio_encoder output padded to [1,512,4,64].
    std::vector<float> audio_feats(kMaxText * kPredFeatFloats, 0.0f);
    std::vector<float> audio_mask (kMaxText, 0.0f);
    const int64_t context_length_scalar = context_length;

    // --- 2. text_prefill → initial hiddens + caches.
    std::vector<float> lm_hidden       (kHidden);
    std::vector<float> residual_hidden (kHidden);
    std::vector<float> prefix_feat_cond(kPredFeatFloats);
    std::vector<float> base_prefill    (kBasePrefillFloats);
    std::vector<float> residual_prefill(kResidualPrefillFloats);
    auto env = LiteRTEngine::get().env();
    {
        auto t_text_tokens = make_type(kLiteRtElementTypeInt64,   {1, kMaxText});
        auto t_text_mask   = make_type(kLiteRtElementTypeFloat32, {1, kMaxText});
        auto t_audio_feats = make_type(kLiteRtElementTypeFloat32,
                                        {1, kMaxText, kPatchSize, kFeatDim});
        auto t_audio_mask  = make_type(kLiteRtElementTypeFloat32, {1, kMaxText});
        auto t_ctxlen      = make_type(kLiteRtElementTypeInt64,   {});

        auto t_lm    = make_type(kLiteRtElementTypeFloat32, {1, kHidden});
        auto t_resid = make_type(kLiteRtElementTypeFloat32, {1, kHidden});
        auto t_pfc   = make_type(kLiteRtElementTypeFloat32, {1, kPatchSize, kFeatDim});
        auto t_bcache = make_type(kLiteRtElementTypeFloat32, {2, 28, 1, 2, 512, 128});
        auto t_rcache = make_type(kLiteRtElementTypeFloat32, {2,  8, 1, 2, 512, 128});

        LiteRtHostBuffer in_tokens (env, t_text_tokens, text_tokens.size() * sizeof(int64_t), text_tokens.data());
        LiteRtHostBuffer in_tmask  (env, t_text_mask,   text_mask.size()   * sizeof(float),   text_mask.data());
        LiteRtHostBuffer in_afeats (env, t_audio_feats, audio_feats.size() * sizeof(float),   audio_feats.data());
        LiteRtHostBuffer in_amask  (env, t_audio_mask,  audio_mask.size()  * sizeof(float),   audio_mask.data());
        LiteRtHostBuffer in_ctxlen (env, t_ctxlen,      sizeof(int64_t),                      &context_length_scalar);

        LiteRtHostBuffer out_lm    (env, t_lm,     lm_hidden.size()        * sizeof(float));
        LiteRtHostBuffer out_resid (env, t_resid,  residual_hidden.size()  * sizeof(float));
        LiteRtHostBuffer out_pfc   (env, t_pfc,    prefix_feat_cond.size() * sizeof(float));
        LiteRtHostBuffer out_bcache(env, t_bcache, base_prefill.size()     * sizeof(float));
        LiteRtHostBuffer out_rcache(env, t_rcache, residual_prefill.size() * sizeof(float));

        LiteRtTensorBuffer ins[5] = {
            in_tokens.raw(), in_tmask.raw(), in_afeats.raw(), in_amask.raw(), in_ctxlen.raw()
        };
        LiteRtTensorBuffer outs[5] = {
            out_lm.raw(), out_resid.raw(), out_pfc.raw(), out_bcache.raw(), out_rcache.raw()
        };
        litert_check(LiteRtRunCompiledModel(text_prefill_compiled_, 0, 5, ins, 5, outs),
                     "text_prefill Run");

        out_lm    .read(lm_hidden.data(),        lm_hidden.size()        * sizeof(float));
        out_resid .read(residual_hidden.data(),  residual_hidden.size()  * sizeof(float));
        out_pfc   .read(prefix_feat_cond.data(), prefix_feat_cond.size() * sizeof(float));
        out_bcache.read(base_prefill.data(),     base_prefill.size()     * sizeof(float));
        out_rcache.read(residual_prefill.data(), residual_prefill.size() * sizeof(float));
    }

    // --- 3. Grow caches 512 → 2560 by zero-padding axis 4. Layout is
    // [2 (K/V), layers, 1 (batch), kv_heads, seq, head_dim]; we copy the first
    // 512 seq slots and leave the remaining 2048 zeroed.
    auto pad_cache = [](const std::vector<float>& src, std::vector<float>& dst,
                        int layers, int kv_heads, int head_dim) {
        constexpr int kPrefSeq = 512;
        constexpr int kFullSeq = 2560;
        std::fill(dst.begin(), dst.end(), 0.0f);
        const size_t per_kv  = static_cast<size_t>(layers) * kv_heads;
        const size_t src_row = static_cast<size_t>(kPrefSeq) * head_dim;
        const size_t dst_row = static_cast<size_t>(kFullSeq) * head_dim;
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
    // and emits one 256-float pred_feat. Every 64 pred_feats are stacked into
    // the decoder's fixed [1, 64, 256] latent slot and streamed out.
    std::vector<float> pred_feat   (kPredFeatFloats);
    std::vector<float> stop_logits (2);
    std::vector<float> next_lm    (kHidden);
    std::vector<float> next_resid (kHidden);
    std::vector<float> noise      (kPredFeatFloats);
    std::vector<float> feature_buffer; feature_buffer.reserve(kDecoderInputFloats);
    std::vector<float> decoder_input(kDecoderInputFloats, 0.0f);
    std::vector<float> pcm         (kDecoderOutputFloats);

    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    auto t_lm     = make_type(kLiteRtElementTypeFloat32, {1, kHidden});
    auto t_resid  = make_type(kLiteRtElementTypeFloat32, {1, kHidden});
    auto t_pfc    = make_type(kLiteRtElementTypeFloat32, {1, kPatchSize, kFeatDim});
    auto t_bcache = make_type(kLiteRtElementTypeFloat32, {2, 28, 1, 2, 2560, 128});
    auto t_rcache = make_type(kLiteRtElementTypeFloat32, {2,  8, 1, 2, 2560, 128});
    auto t_pos    = make_type(kLiteRtElementTypeInt64,   {});
    auto t_noise  = make_type(kLiteRtElementTypeFloat32, {1, kFeatDim, kPatchSize});
    auto t_pred   = make_type(kLiteRtElementTypeFloat32, {1, kPatchSize, kFeatDim});
    auto t_stop   = make_type(kLiteRtElementTypeFloat32, {1, 2});
    auto t_dec_in = make_type(kLiteRtElementTypeFloat32, {1, kFramesPerChunk, kPredFeatFloats});
    auto t_pcm    = make_type(kLiteRtElementTypeFloat32, {1, kDecoderOutputFloats});

    auto flush_decoder = [&](size_t valid_steps, bool is_final) {
        std::fill(decoder_input.begin(), decoder_input.end(), 0.0f);
        // feature_buffer is N consecutive pred_feat tensors, each
        // [patch_size=4, feat_dim=64] in row-major order (= pred_feat[p][c]
        // at offset p*64+c).
        //
        // Decoder input is [1, feat_dim=64, frames=256] in row-major =
        // latent[c][t] at offset c*256+t. Each AR step occupies 4 frames
        // (patch_size); the model reconstructs audio by spreading the patch
        // values across consecutive output frames.
        //
        // Mapping: latent[c, step*4 + p] = pred_feat[step][p][c]
        const size_t cap_steps = std::min<size_t>(
            valid_steps,
            static_cast<size_t>(kFramesPerChunk));  // 64 AR steps fills the window
        for (size_t step_idx = 0; step_idx < cap_steps; ++step_idx) {
            const float* pred = feature_buffer.data() + step_idx * kPredFeatFloats;
            for (int p = 0; p < kPatchSize; ++p) {
                const int t = static_cast<int>(step_idx) * kPatchSize + p;
                if (t >= kDecoderInputFloats / kFeatDim) break;
                for (int c = 0; c < kFeatDim; ++c) {
                    decoder_input[c * (kDecoderInputFloats / kFeatDim) + t]
                        = pred[p * kFeatDim + c];
                }
            }
        }

        LiteRtHostBuffer in_latent(env, t_dec_in,
                                   decoder_input.size() * sizeof(float),
                                   decoder_input.data());
        LiteRtHostBuffer out_pcm(env, t_pcm, pcm.size() * sizeof(float));
        LiteRtTensorBuffer ins[1]  = { in_latent.raw() };
        LiteRtTensorBuffer outs[1] = { out_pcm.raw() };
        litert_check(LiteRtRunCompiledModel(audio_decoder_compiled_, 0, 1, ins, 1, outs),
                     "audio_decoder Run");

        out_pcm.read(pcm.data(), pcm.size() * sizeof(float));

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

        LiteRtHostBuffer in_lm    (env, t_lm,     lm_hidden.size()        * sizeof(float), lm_hidden.data());
        LiteRtHostBuffer in_resid (env, t_resid,  residual_hidden.size()  * sizeof(float), residual_hidden.data());
        LiteRtHostBuffer in_pfc   (env, t_pfc,    prefix_feat_cond.size() * sizeof(float), prefix_feat_cond.data());
        LiteRtHostBuffer in_bcache(env, t_bcache, base_cache.size()       * sizeof(float), base_cache.data());
        LiteRtHostBuffer in_rcache(env, t_rcache, residual_cache.size()   * sizeof(float), residual_cache.data());
        LiteRtHostBuffer in_pos   (env, t_pos,    sizeof(int64_t),                         &position_id);
        LiteRtHostBuffer in_noise (env, t_noise,  noise.size()            * sizeof(float), noise.data());

        LiteRtHostBuffer out_pred  (env, t_pred,   pred_feat.size()      * sizeof(float));
        LiteRtHostBuffer out_stop  (env, t_stop,   stop_logits.size()    * sizeof(float));
        LiteRtHostBuffer out_lm    (env, t_lm,     next_lm.size()        * sizeof(float));
        LiteRtHostBuffer out_resid (env, t_resid,  next_resid.size()     * sizeof(float));
        LiteRtHostBuffer out_bcache(env, t_bcache, base_cache.size()     * sizeof(float));
        LiteRtHostBuffer out_rcache(env, t_rcache, residual_cache.size() * sizeof(float));

        LiteRtTensorBuffer ins[7] = {
            in_lm.raw(), in_resid.raw(), in_pfc.raw(),
            in_bcache.raw(), in_rcache.raw(), in_pos.raw(), in_noise.raw()
        };
        LiteRtTensorBuffer outs[6] = {
            out_pred.raw(), out_stop.raw(), out_lm.raw(), out_resid.raw(),
            out_bcache.raw(), out_rcache.raw()
        };
        litert_check(LiteRtRunCompiledModel(token_step_compiled_, 0, 7, ins, 6, outs),
                     "token_step Run");

        out_pred  .read(pred_feat.data(),      pred_feat.size()      * sizeof(float));
        out_stop  .read(stop_logits.data(),    stop_logits.size()    * sizeof(float));
        out_lm    .read(next_lm.data(),        next_lm.size()        * sizeof(float));
        out_resid .read(next_resid.data(),     next_resid.size()     * sizeof(float));
        out_bcache.read(base_cache.data(),     base_cache.size()     * sizeof(float));
        out_rcache.read(residual_cache.data(), residual_cache.size() * sizeof(float));

        // Roll forward — the next step consumes the freshest hiddens, and
        // prefix_feat_cond is *replaced* by the latest pred_feat (the graph
        // reuses that slot as the AR conditioning signal).
        lm_hidden        = next_lm;
        residual_hidden  = next_resid;
        prefix_feat_cond = pred_feat;

        feature_buffer.insert(feature_buffer.end(), pred_feat.begin(), pred_feat.end());
        ++steps_done;
        ++steps_in_chunk;

        const bool stop_signal = stop_logits[1] > stop_logits[0]
                              && steps_done > min_stop_steps_;
        if (stop_signal) stopped_by_model = true;

        if (steps_in_chunk == kFramesPerChunk) {
            flush_decoder(kFramesPerChunk, /*is_final=*/false);
            steps_in_chunk = 0;
        }
        if (stop_signal) break;
    }

    if (steps_in_chunk > 0) {
        flush_decoder(static_cast<size_t>(steps_in_chunk), /*is_final=*/true);
    } else {
        on_chunk(nullptr, 0, /*is_final=*/true);
    }
    (void)stopped_by_model;  // informational only
}

}  // namespace speech_core
