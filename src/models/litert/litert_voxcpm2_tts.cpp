#include "speech_core/models/litert_voxcpm2_tts.h"

#include "speech_core/audio/resampler.h"
#include "speech_core/models/voxcpm2_prompt.h"
#include "tts_postprocess_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace speech_core {

namespace {

// Read the prefill context window (max_text_tokens) from the loaded graph: its
// audio_feats input is the only rank-4 tensor, shaped [1, max_text, patch,
// feat]. Lets a 256- or 512-token export both work. Falls back to 512 (the
// deployed bundle) if introspection isn't available.
int query_prefill_context(LiteRtModel model) {
    constexpr int kDefault = 512;
    if (!model) return kDefault;
    LiteRtParamIndex main_idx = 0;
    if (LiteRtGetMainModelSubgraphIndex(model, &main_idx) != kLiteRtStatusOk) return kDefault;
    LiteRtSubgraph sg = nullptr;
    if (LiteRtGetModelSubgraph(model, main_idx, &sg) != kLiteRtStatusOk) return kDefault;
    LiteRtParamIndex n = 0;
    if (LiteRtGetNumSubgraphInputs(sg, &n) != kLiteRtStatusOk) return kDefault;
    for (LiteRtParamIndex i = 0; i < n; ++i) {
        LiteRtTensor t = nullptr;
        if (LiteRtGetSubgraphInput(sg, i, &t) != kLiteRtStatusOk) continue;
        LiteRtRankedTensorType rt{};
        if (LiteRtGetRankedTensorType(t, &rt) != kLiteRtStatusOk) continue;
        if (rt.layout.rank == 4) {  // audio_feats: [1, max_text, patch, feat]
            const int dim = static_cast<int>(rt.layout.dimensions[1]);
            if (dim > 0) return dim;
        }
    }
    return kDefault;
}

void validate_synthesis_options(const VoxCPM2SynthesisOptions& options) {
    validate_tts_synthesis_options(options, "VoxCPM2");
}

std::vector<float> apply_postprocess(
    std::vector<float> audio,
    int sample_rate,
    VoxCPM2PostProcessFlags flags) {
    return internal::apply_tts_postprocess_owned(std::move(audio), sample_rate, flags);
}

}  // namespace

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
    // Remember the prefill path + hw_accel choice for the lazy load/unload
    // cycle that runs across every synthesize() call. The other three graphs
    // (token_step, audio_encoder, audio_decoder) stay resident across calls
    // because they're either used per-step in the AR loop (token_step) or
    // cheap enough that the reload-each-call latency would dominate.
    text_prefill_path_     = text_prefill_path;
    text_prefill_hw_accel_ = hw_accel;

    auto& engine = LiteRTEngine::get();
    engine.load(token_step_path,    hw_accel, &token_step_model_,    &token_step_compiled_);
    engine.load(audio_encoder_path, hw_accel, &audio_encoder_model_, &audio_encoder_compiled_);
    engine.load(audio_decoder_path, hw_accel, &audio_decoder_model_, &audio_decoder_compiled_);

    // Load text_prefill once so we can derive the context geometry from its
    // input shape, then release it. The engine's retained_buffers_ entry is
    // freed too -- otherwise the 1.9 GiB file would sit resident even though
    // the compiled model is gone. The first synthesize() call pays the
    // reload cost; subsequent calls pay it again because we release at the
    // end of each prefill (worker-side amortisation lives outside this
    // class).
    load_text_prefill();
    max_text_                = query_prefill_context(text_prefill_model_);
    full_seq_                = max_text_ + kMaxGenerated;
    base_cache_floats_       = 2L * kBaseLayers     * kKvHeads * full_seq_ * kHeadDim;
    residual_cache_floats_   = 2L * kResidualLayers * kKvHeads * full_seq_ * kHeadDim;
    base_prefill_floats_     = 2L * kBaseLayers     * kKvHeads * max_text_ * kHeadDim;
    residual_prefill_floats_ = 2L * kResidualLayers * kKvHeads * max_text_ * kHeadDim;
    release_text_prefill();

    tokenizer_         = std::make_unique<VoxCPM2Tokenizer>(tokenizer_path);
    audio_start_token_ = tokenizer_->token_id("<|audio_start|>");
    if (audio_start_token_ < 0) {
        throw std::runtime_error(
            "LiteRT VoxCPM2: tokenizer is missing <|audio_start|> — bundle is malformed");
    }
    // Reference boundary tokens are only needed for cloning; absence is not
    // fatal here (set_reference() throws if they're missing when used).
    ref_audio_start_token_ = tokenizer_->token_id("<|ref_audio_start|>");
    ref_audio_end_token_   = tokenizer_->token_id("<|ref_audio_end|>");

    // Start the keep-warm reaper. The thread polls every second; when
    // idle_release_ms_ is 0 (the default) its eviction step is a no-op so
    // the steady-state behaviour matches the original lazy-load path. A
    // caller flips it on via set_idle_release_ms() any time after construct.
    reaper_thread_ = std::thread(&LiteRTVoxCPM2Tts::reaper_loop_, this);
}

LiteRTVoxCPM2Tts::~LiteRTVoxCPM2Tts() {
    // Shut the reaper down first so it can't fire after we destroy the
    // prefill graph (or any of the resident graphs it could observe).
    {
        std::lock_guard<std::mutex> lock(reaper_mutex_);
        reaper_stop_.store(true);
    }
    reaper_cv_.notify_all();
    if (reaper_thread_.joinable()) reaper_thread_.join();

    if (audio_decoder_compiled_) LiteRtDestroyCompiledModel(audio_decoder_compiled_);
    if (audio_decoder_model_)    LiteRtDestroyModel(audio_decoder_model_);
    if (audio_encoder_compiled_) LiteRtDestroyCompiledModel(audio_encoder_compiled_);
    if (audio_encoder_model_)    LiteRtDestroyModel(audio_encoder_model_);
    if (token_step_compiled_)    LiteRtDestroyCompiledModel(token_step_compiled_);
    if (token_step_model_)       LiteRtDestroyModel(token_step_model_);
    release_text_prefill();
}

void LiteRTVoxCPM2Tts::set_idle_release_ms(std::int64_t ms) {
    idle_release_ms_.store(ms);
}

void LiteRTVoxCPM2Tts::reaper_loop_() {
    constexpr auto poll_interval = std::chrono::seconds(1);
    while (true) {
        {
            std::unique_lock<std::mutex> lock(reaper_mutex_);
            reaper_cv_.wait_for(lock, poll_interval,
                                [this] { return reaper_stop_.load(); });
        }
        if (reaper_stop_.load()) return;

        const std::int64_t idle_ms = idle_release_ms_.load();
        if (idle_ms <= 0) continue;

        // Don't block synthesize() -- if a call is in progress (holding the
        // prefill mutex), defer eviction to the next tick.
        std::unique_lock<std::mutex> lock(prefill_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) continue;

        if (!text_prefill_compiled_) continue;  // already released

        const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - last_prefill_used_)
                            .count();
        if (age > idle_ms) {
            release_text_prefill();
        }
    }
}

void LiteRTVoxCPM2Tts::load_text_prefill() {
    if (text_prefill_compiled_) return;  // already loaded
    LiteRTEngine::get().load(text_prefill_path_, text_prefill_hw_accel_,
                              &text_prefill_model_, &text_prefill_compiled_);
}

void LiteRTVoxCPM2Tts::release_text_prefill() {
    if (text_prefill_compiled_) {
        LiteRtDestroyCompiledModel(text_prefill_compiled_);
        text_prefill_compiled_ = nullptr;
    }
    if (text_prefill_model_) {
        LiteRtDestroyModel(text_prefill_model_);
        text_prefill_model_ = nullptr;
    }
    // Engine's retained_buffers_ holds a ~1.9 GiB copy of the file (kept
    // because LiteRtCreateModelFromBuffer's input must outlive the model);
    // drop it now that the model is destroyed.
    LiteRTEngine::get().release_buffer(text_prefill_path_);
}

void LiteRTVoxCPM2Tts::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Reference conditioning — run audio_encoder on a speaker clip and cache the
// encoded latents for subsequent synthesize() calls.
// ---------------------------------------------------------------------------

void LiteRTVoxCPM2Tts::clear_reference() {
    ref_feats_.clear();
    ref_feats_.shrink_to_fit();
    ref_frames_ = 0;
}

void LiteRTVoxCPM2Tts::set_reference(const float* pcm, size_t length, int sample_rate) {
    clear_reference();
    if (!pcm || length == 0) return;
    if (ref_audio_start_token_ < 0 || ref_audio_end_token_ < 0) {
        throw std::runtime_error(
            "LiteRT VoxCPM2: tokenizer lacks <|ref_audio_start|>/<|ref_audio_end|> — "
            "this bundle does not support voice cloning");
    }

    // Resample the reference to the encoder's conditioning rate (16 kHz).
    std::vector<float> mono;
    if (sample_rate != kCondSampleRate) {
        mono = Resampler::resample(pcm, length, sample_rate, kCondSampleRate);
    } else {
        mono.assign(pcm, pcm + length);
    }
    if (mono.empty()) return;

    // Trim leading silence. The encoder window is a fixed 6.4 s taken from the
    // front, so a quiet lead-in (common in user clips) would otherwise fill it
    // with silence and clone a silent speaker. Gate on a fraction of the clip's
    // peak so quiet-but-real speech isn't trimmed; keep a 100 ms pre-roll. If
    // the clip is entirely below the gate, leave it as-is.
    {
        float peak = 0.0f;
        for (float v : mono) peak = std::max(peak, std::abs(v));
        const float gate = std::max(0.01f, 0.1f * peak);
        constexpr int kWin = 320;  // 20 ms @ 16 kHz
        size_t start = 0;
        for (; start + kWin <= mono.size(); start += kWin) {
            double ss = 0.0;
            for (int i = 0; i < kWin; ++i) ss += double(mono[start + i]) * mono[start + i];
            if (std::sqrt(ss / kWin) > gate) break;
        }
        if (start + kWin <= mono.size() && start > 0) {
            const size_t preroll = std::min<size_t>(start, kCondSampleRate / 10);
            mono.erase(mono.begin(), mono.begin() + (start - preroll));
        }
    }

    // Rescue quiet reference clips. VoxCPM2's audio_encoder conditions on
    // both timbre AND amplitude — a reference at RMS ~0.002 (e.g. an
    // unmastered phone-mic capture) clones at sub-audible level, because
    // output amplitude tracks reference amplitude within ±10%. The upstream
    // openbmb VoxCPM2 Python (src/voxcpm/model/voxcpm2.py:_encode_wav)
    // similarly does no amplitude normalization — it relies on callers
    // passing already-loudness-normalized reference clips. We do the same for
    // any clip already in a healthy range (RMS ≥ kQuietThreshold), but
    // rescue obviously-quiet inputs by scaling them to ~kTargetRms with a
    // peak cap. Pure scalar gain; no compression / EQ.
    {
        double ss = 0.0;
        float peak = 0.0f;
        for (float v : mono) {
            ss += double(v) * v;
            peak = std::max(peak, std::abs(v));
        }
        if (!mono.empty() && peak > 1e-6f) {
            const double rms = std::sqrt(ss / mono.size());
            const float kQuietThreshold = 0.04f;   // upstream-healthy refs are above this
            const float kTargetRms      = 0.08f;
            const float kPeakCap        = 0.95f;
            if (rms < kQuietThreshold) {
                float scale = static_cast<float>(kTargetRms / rms);
                if (peak * scale > kPeakCap) scale = kPeakCap / peak;
                for (float& v : mono) v *= scale;
            }
        }
    }

    // Frames backed by *real* (pre-pad) audio. The encoder always emits
    // kMaxRefFrames, but trailing frames over zero-padding carry silence — we
    // condition only on the real ones (matches the MLX dynamic-length path).
    const size_t real_samples = std::min<size_t>(mono.size(), kRefAudioSamples);
    int real_frames = static_cast<int>((real_samples + kRefFrameStride - 1) / kRefFrameStride);
    real_frames = std::clamp(real_frames, 1, kMaxRefFrames);

    // The audio_encoder graph has a fixed-length input — pad/truncate to it.
    mono.resize(kRefAudioSamples, 0.0f);

    auto env = LiteRTEngine::get().env();
    auto t_in  = make_type(kLiteRtElementTypeFloat32, {1, kRefAudioSamples});
    auto t_out = make_type(kLiteRtElementTypeFloat32,
                           {1, kMaxRefFrames, kPatchSize, kFeatDim});

    LiteRtHostBuffer in_audio(env, t_in, mono.size() * sizeof(float), mono.data());
    std::vector<float> enc_out(static_cast<size_t>(kMaxRefFrames) * kPredFeatFloats);
    LiteRtHostBuffer out_feats(env, t_out, enc_out.size() * sizeof(float));

    LiteRtTensorBuffer ins[1]  = { in_audio.raw() };
    LiteRtTensorBuffer outs[1] = { out_feats.raw() };
    litert_check(LiteRtRunCompiledModel(audio_encoder_compiled_, 0, 1, ins, 1, outs),
                 "audio_encoder Run");
    out_feats.read(enc_out.data(), enc_out.size() * sizeof(float));

    // Keep the first real_frames frames; each is one audio_feats slot.
    ref_feats_.assign(enc_out.begin(),
                      enc_out.begin() + static_cast<size_t>(real_frames) * kPredFeatFloats);
    ref_frames_ = real_frames;

    if (std::getenv("VOXCPM2_DEBUG")) {
        double ss = 0.0;
        for (float v : ref_feats_) ss += static_cast<double>(v) * v;
        const double rms = ref_feats_.empty() ? 0.0 : std::sqrt(ss / ref_feats_.size());
        LOGI("VoxCPM2 reference: %d frames (%zu real samples), feats rms=%.4f "
             "[tokens: audio_start=%d ref_start=%d ref_end=%d]",
             ref_frames_, real_samples, rms,
             audio_start_token_, ref_audio_start_token_, ref_audio_end_token_);
    }
}

// ---------------------------------------------------------------------------
// synthesize() — mirrors speech-models/.../smoke_litert_roundtrip.py
// ---------------------------------------------------------------------------

void LiteRTVoxCPM2Tts::synthesize(const std::string& text,
                                   const std::string& language,
                                   TTSChunkCallback on_chunk)
{
    synthesize_with_options(text, language, VoxCPM2SynthesisOptions{}, std::move(on_chunk));
}

void LiteRTVoxCPM2Tts::synthesize_with_options(const std::string& text,
                                                const std::string& /*language*/,
                                                const VoxCPM2SynthesisOptions& options,
                                                TTSChunkCallback on_chunk)
{
    if (!on_chunk) return;
    validate_synthesis_options(options);
    cancelled_.store(false, std::memory_order_relaxed);

    // Reset per-call metadata; populated as the AR loop runs / completes.
    tokens_generated_      = 0;
    stopped_on_stop_token_ = false;
    seed_used_             = 0;

    // --- 1. Build the prefill sequence. The target text is either the raw
    // text, or "({instruction}){text}" when an instruction is present,
    // followed by <|audio_start|> (the cue to begin generating audio). When a
    // reference clip is set, a
    // [<|ref_audio_start|>, latents…, <|ref_audio_end|>] block is prepended and
    // its latents fill audio_feats (audio_mask=1), conditioning the output on
    // the reference speaker. Mirrors the MLX ICL clone path in speech-swift.
    std::string prompt = format_voxcpm2_prompt(text, instruction_);
    std::vector<int> target_ids = tokenizer_->encode(prompt);
    target_ids.push_back(audio_start_token_);

    // With a reference block, the target text must not carry the leading BOS
    // that encode() prepends: the model was trained on
    // [ref_block, text, <|audio_start|>] with no sentence-start marker after
    // the reference. A stray <s> there makes it stop early and emit silence.
    // (The MLX clone path tokenizes without a BOS for the same reason.)
    if (ref_frames_ > 0 && !target_ids.empty()
        && target_ids.front() == tokenizer_->bos_id()) {
        target_ids.erase(target_ids.begin());
    }

    std::vector<int64_t> text_tokens(max_text_, 0);
    std::vector<float>   text_mask  (max_text_, 0.0f);
    std::vector<float>   audio_feats(static_cast<size_t>(max_text_) * kPredFeatFloats, 0.0f);
    std::vector<float>   audio_mask (max_text_, 0.0f);

    int pos = 0;
    if (ref_frames_ > 0) {
        // Reserve room for the ref block plus at least one target token.
        if (2 + ref_frames_ + 1 > max_text_) {
            throw std::runtime_error(
                "LiteRT VoxCPM2: reference block does not fit the context window");
        }
        text_tokens[pos] = ref_audio_start_token_;
        text_mask[pos]   = 1.0f;
        ++pos;
        for (int f = 0; f < ref_frames_; ++f) {
            audio_mask[pos] = 1.0f;
            std::memcpy(audio_feats.data() + static_cast<size_t>(pos) * kPredFeatFloats,
                        ref_feats_.data()  + static_cast<size_t>(f)   * kPredFeatFloats,
                        kPredFeatFloats * sizeof(float));
            ++pos;
        }
        text_tokens[pos] = ref_audio_end_token_;
        text_mask[pos]   = 1.0f;
        ++pos;
    }

    // Target text fills the rest of the window. If it would overflow, drop from
    // the front but always keep the trailing <|audio_start|> cue.
    const int avail = max_text_ - pos;
    if (static_cast<int>(target_ids.size()) > avail) {
        std::vector<int> trimmed;
        trimmed.reserve(static_cast<size_t>(avail));
        for (int i = 0; i < avail - 1; ++i) trimmed.push_back(target_ids[i]);
        trimmed.push_back(audio_start_token_);
        target_ids.swap(trimmed);
    }
    for (int id : target_ids) {
        text_tokens[pos] = id;
        text_mask[pos]   = 1.0f;
        ++pos;
    }
    const int     context_length        = pos;
    const int64_t context_length_scalar = context_length;

    // --- 2. text_prefill → initial hiddens + caches.
    //
    // Lifecycle is controlled by idle_release_ms_:
    //   - 0 (default): load on entry, release on exit. The classic lazy-load
    //     path. Every call pays the cold-load (~3-5 s) but cold-RSS sits at
    //     ~6 GiB between calls.
    //   - >0 (keep-warm): load on entry if not already resident, leave loaded
    //     on exit. The reaper thread releases after the configured idle
    //     window. Sustains ~9-10 GiB warm RSS but zero cold-load on the hot
    //     path -- the right trade for LiveKit/realtime turns arriving within
    //     seconds of each other.
    //
    // The entire prefill stage runs under prefill_mutex_ so the reaper can't
    // evict the graph mid-Run(). The AR loop below uses token_step (a
    // separate graph) and doesn't need the lock.
    std::vector<float> lm_hidden       (kHidden);
    std::vector<float> residual_hidden (kHidden);
    std::vector<float> prefix_feat_cond(kPredFeatFloats);
    std::vector<float> base_prefill    (base_prefill_floats_);
    std::vector<float> residual_prefill(residual_prefill_floats_);
    auto env = LiteRTEngine::get().env();
    std::unique_lock<std::mutex> prefill_lock(prefill_mutex_);
    load_text_prefill();
    last_prefill_used_ = std::chrono::steady_clock::now();
    {
        auto t_text_tokens = make_type(kLiteRtElementTypeInt64,   {1, max_text_});
        auto t_text_mask   = make_type(kLiteRtElementTypeFloat32, {1, max_text_});
        auto t_audio_feats = make_type(kLiteRtElementTypeFloat32,
                                        {1, max_text_, kPatchSize, kFeatDim});
        auto t_audio_mask  = make_type(kLiteRtElementTypeFloat32, {1, max_text_});
        auto t_ctxlen      = make_type(kLiteRtElementTypeInt64,   {});

        auto t_lm    = make_type(kLiteRtElementTypeFloat32, {1, kHidden});
        auto t_resid = make_type(kLiteRtElementTypeFloat32, {1, kHidden});
        auto t_pfc   = make_type(kLiteRtElementTypeFloat32, {1, kPatchSize, kFeatDim});
        auto t_bcache = make_type(kLiteRtElementTypeFloat32,
                                  {2, kBaseLayers,     1, kKvHeads, max_text_, kHeadDim});
        auto t_rcache = make_type(kLiteRtElementTypeFloat32,
                                  {2, kResidualLayers, 1, kKvHeads, max_text_, kHeadDim});

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
    // Release strategy: depends on idle_release_ms_.
    //
    // When 0 (always-release path): free the graph + retained_buffers_ entry
    // immediately -- the original behaviour that fits the AR loop inside the
    // 11 GiB pod limit on the prod CCX23.
    //
    // When >0 (keep-warm path): leave the graph loaded and just update
    // last_prefill_used_; the reaper thread evicts after the idle window.
    // Glibc's heap arena retains the bookkeeping pages either way, so the
    // steady-state RSS is roughly the same once the first synth has run --
    // the win is purely on cold-load latency for back-to-back calls.
    last_prefill_used_ = std::chrono::steady_clock::now();
    if (idle_release_ms_.load() == 0) {
        release_text_prefill();
    }
    prefill_lock.unlock();

    if (std::getenv("VOXCPM2_DEBUG")) {
        auto rms = [](const std::vector<float>& v) {
            double s = 0.0; for (float x : v) s += double(x) * x;
            return v.empty() ? 0.0 : std::sqrt(s / v.size());
        };
        LOGI("VoxCPM2 prefill: ctx_len=%d ref_frames=%d lm_rms=%.4f resid_rms=%.4f pfc_rms=%.4f",
             context_length, ref_frames_, rms(lm_hidden), rms(residual_hidden),
             rms(prefix_feat_cond));
    }

    // --- 3. Grow caches from the prefill window (max_text_) to the token_step
    // window (full_seq_ = max_text_ + max_generated) by zero-padding axis 4.
    // Layout is [2 (K/V), layers, 1 (batch), kv_heads, seq, head_dim]; copy the
    // first max_text_ seq slots and leave the remaining slots zeroed.
    auto pad_cache = [pref_seq = max_text_, full_seq = full_seq_](
                         const std::vector<float>& src, std::vector<float>& dst,
                         int layers, int kv_heads, int head_dim) {
        std::fill(dst.begin(), dst.end(), 0.0f);
        const size_t per_kv  = static_cast<size_t>(layers) * kv_heads;
        const size_t src_row = static_cast<size_t>(pref_seq) * head_dim;
        const size_t dst_row = static_cast<size_t>(full_seq) * head_dim;
        for (int kv = 0; kv < 2; ++kv) {
            for (size_t i = 0; i < per_kv; ++i) {
                const float* src_ptr = src.data() + (kv * per_kv + i) * src_row;
                float*       dst_ptr = dst.data() + (kv * per_kv + i) * dst_row;
                std::memcpy(dst_ptr, src_ptr, src_row * sizeof(float));
            }
        }
    };
    std::vector<float> base_cache    (base_cache_floats_);
    std::vector<float> residual_cache(residual_cache_floats_);
    pad_cache(base_prefill,     base_cache,     kBaseLayers,     kKvHeads, kHeadDim);
    pad_cache(residual_prefill, residual_cache, kResidualLayers, kKvHeads, kHeadDim);
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

    // The token_step graph is a diffusion-style sampler: it consumes a fresh
    // Gaussian noise tensor each step, so the seed below genuinely controls the
    // output. seed_ == 0 means "pick a non-deterministic seed"; either way the
    // resolved value is recorded for seed_used().
    uint32_t rng_seed = seed_;
    if (rng_seed == 0) {
        std::random_device rd;
        rng_seed = rd();
    }
    seed_used_ = rng_seed;
    std::mt19937 rng(rng_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    auto t_lm     = make_type(kLiteRtElementTypeFloat32, {1, kHidden});
    auto t_resid  = make_type(kLiteRtElementTypeFloat32, {1, kHidden});
    auto t_pfc    = make_type(kLiteRtElementTypeFloat32, {1, kPatchSize, kFeatDim});
    auto t_bcache = make_type(kLiteRtElementTypeFloat32,
                              {2, kBaseLayers,     1, kKvHeads, full_seq_, kHeadDim});
    auto t_rcache = make_type(kLiteRtElementTypeFloat32,
                              {2, kResidualLayers, 1, kKvHeads, full_seq_, kHeadDim});
    auto t_pos    = make_type(kLiteRtElementTypeInt64,   {});
    auto t_noise  = make_type(kLiteRtElementTypeFloat32, {1, kFeatDim, kPatchSize});
    auto t_pred   = make_type(kLiteRtElementTypeFloat32, {1, kPatchSize, kFeatDim});
    auto t_stop   = make_type(kLiteRtElementTypeFloat32, {1, 2});
    auto t_dec_in = make_type(kLiteRtElementTypeFloat32, {1, kFramesPerChunk, kPredFeatFloats});
    auto t_pcm    = make_type(kLiteRtElementTypeFloat32, {1, kDecoderOutputFloats});

    std::vector<float> synthesized_pcm;

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

        if (valid_samples > 0) {
            if (options.mode == VoxCPM2SynthesisMode::Buffered) {
                synthesized_pcm.insert(
                    synthesized_pcm.end(),
                    pcm.begin(),
                    pcm.begin() + static_cast<std::ptrdiff_t>(valid_samples));
            } else {
                on_chunk(pcm.data(), valid_samples, is_final);
            }
        }

        feature_buffer.clear();
    };

    int  steps_done       = 0;
    int  steps_in_chunk   = 0;
    bool stopped_by_model = false;
    const bool debug_steps = std::getenv("VOXCPM2_DEBUG") != nullptr;

    // Pre-allocate the token_step input + output tensor buffers ONCE before
    // the AR loop. The graph shape is fixed (full_seq_ + kHidden + ... are
    // constants across iterations), so the buffer sizes don't change step
    // to step. The original implementation allocated 13 fresh managed
    // tensor buffers PER step × ~1000-2000 steps per synthesize, pushing
    // RSS past 14 GiB mid-call and OOMKilling the synth pod on the 16 GiB
    // CCX23. With buffers hoisted, allocation count drops to 13 total
    // (per call), each reused via write()/read() per step.
    //
    // The data backing each LiteRtHostBuffer is aligned-managed memory
    // owned by LiteRT (LiteRtCreateManagedTensorBuffer); the constructor
    // without a `seed` argument allocates fresh but leaves the contents
    // uninitialised. We seed them inside the loop via write().
    LiteRtHostBuffer in_lm    (env, t_lm,     lm_hidden.size()        * sizeof(float));
    LiteRtHostBuffer in_resid (env, t_resid,  residual_hidden.size()  * sizeof(float));
    LiteRtHostBuffer in_pfc   (env, t_pfc,    prefix_feat_cond.size() * sizeof(float));
    LiteRtHostBuffer in_bcache(env, t_bcache, base_cache.size()       * sizeof(float));
    LiteRtHostBuffer in_rcache(env, t_rcache, residual_cache.size()   * sizeof(float));
    LiteRtHostBuffer in_pos   (env, t_pos,    sizeof(int64_t));
    LiteRtHostBuffer in_noise (env, t_noise,  noise.size()            * sizeof(float));

    LiteRtHostBuffer out_pred  (env, t_pred,   pred_feat.size()      * sizeof(float));
    LiteRtHostBuffer out_stop  (env, t_stop,   stop_logits.size()    * sizeof(float));
    LiteRtHostBuffer out_lm    (env, t_lm,     next_lm.size()        * sizeof(float));
    LiteRtHostBuffer out_resid (env, t_resid,  next_resid.size()     * sizeof(float));
    LiteRtHostBuffer out_bcache(env, t_bcache, base_cache.size()     * sizeof(float));
    LiteRtHostBuffer out_rcache(env, t_rcache, residual_cache.size() * sizeof(float));

    // ins/outs are tiny (just pointers). Stays out of the loop too so we
    // don't redo the array fill every step. LiteRtRunCompiledModel reads
    // these by pointer; the underlying TensorBuffer handles are stable
    // for the buffer's lifetime.
    LiteRtTensorBuffer ins[7] = {
        in_lm.raw(), in_resid.raw(), in_pfc.raw(),
        in_bcache.raw(), in_rcache.raw(), in_pos.raw(), in_noise.raw()
    };
    LiteRtTensorBuffer outs[6] = {
        out_pred.raw(), out_stop.raw(), out_lm.raw(), out_resid.raw(),
        out_bcache.raw(), out_rcache.raw()
    };

    for (int step = 0; step < max_steps_; ++step) {
        if (cancelled_.load(std::memory_order_relaxed)) break;

        for (float& v : noise) v = normal(rng);
        const int64_t position_id = static_cast<int64_t>(context_length + step);

        // Refresh each input buffer with this step's data. write() is a
        // Lock-memcpy-Unlock on LiteRT's aligned host memory — no extra
        // allocation, just copies the bytes the graph will consume.
        in_lm    .write(lm_hidden.data(),        lm_hidden.size()        * sizeof(float));
        in_resid .write(residual_hidden.data(),  residual_hidden.size()  * sizeof(float));
        in_pfc   .write(prefix_feat_cond.data(), prefix_feat_cond.size() * sizeof(float));
        in_bcache.write(base_cache.data(),       base_cache.size()       * sizeof(float));
        in_rcache.write(residual_cache.data(),   residual_cache.size()   * sizeof(float));
        in_pos   .write(&position_id,            sizeof(int64_t));
        in_noise .write(noise.data(),            noise.size()            * sizeof(float));

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

        const bool stop_signal = stop_on_stop_token_
                              && stop_logits[1] > stop_logits[0]
                              && steps_done > min_stop_steps_;
        if (stop_signal) stopped_by_model = true;

        if (debug_steps && (step < 6 || stop_signal)) {
            float pmax = 0.0f; double sum = 0.0, sq = 0.0;
            for (float v : pred_feat) { pmax = std::max(pmax, std::abs(v)); sum += v; sq += double(v) * v; }
            const double mean = sum / pred_feat.size();
            const double sd = std::sqrt(std::max(0.0, sq / pred_feat.size() - mean * mean));
            LOGI("VoxCPM2 step %d: pred|max|=%.4f mean=%.4f std=%.4f stop=[%.2f,%.2f]%s",
                 step, pmax, mean, sd, stop_logits[0], stop_logits[1],
                 stop_signal ? " STOP" : "");
        }

        if (steps_in_chunk == kFramesPerChunk) {
            flush_decoder(kFramesPerChunk, /*is_final=*/false);
            steps_in_chunk = 0;
        }
        if (stop_signal) break;
    }

    if (steps_in_chunk > 0) {
        flush_decoder(static_cast<size_t>(steps_in_chunk), /*is_final=*/true);
    } else if (options.mode == VoxCPM2SynthesisMode::Streaming) {
        on_chunk(nullptr, 0, /*is_final=*/true);
    }

    if (options.mode == VoxCPM2SynthesisMode::Buffered) {
        if (!synthesized_pcm.empty()) {
            std::vector<float> processed_pcm = apply_postprocess(
                std::move(synthesized_pcm),
                output_sample_rate(),
                options.postprocess_flags);
            on_chunk(processed_pcm.data(), processed_pcm.size(), /*is_final=*/true);
        } else {
            on_chunk(nullptr, 0, /*is_final=*/true);
        }
    }

    // Publish synthesis metadata for the getters (the streaming API can't
    // return it inline). tokens_generated == AR steps actually emitted;
    // stopped_on_stop_token distinguishes a model-driven stop from hitting the
    // max_steps budget or a cancel().
    tokens_generated_      = steps_done;
    stopped_on_stop_token_ = stopped_by_model;
}

}  // namespace speech_core
