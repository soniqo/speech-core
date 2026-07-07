#include "speech_core/models/litert_parakeet_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/models/parakeet_language_guidance.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace speech_core {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

LiteRTParakeetStt::LiteRTParakeetStt(const std::string& encoder_path,
                                      const std::string& decoder_joint_path,
                                      const std::string& vocab_path,
                                      bool hw_accel)
{
    auto& engine = LiteRTEngine::get();
    engine.load(encoder_path,       hw_accel, &enc_model_, &enc_compiled_);
    engine.load(decoder_joint_path, false,    &dec_model_, &dec_compiled_);

    load_vocab(vocab_path);
}

LiteRTParakeetStt::~LiteRTParakeetStt() {
    if (dec_compiled_) LiteRtDestroyCompiledModel(dec_compiled_);
    if (dec_model_)    LiteRtDestroyModel(dec_model_);
    if (enc_compiled_) LiteRtDestroyCompiledModel(enc_compiled_);
    if (enc_model_)    LiteRtDestroyModel(enc_model_);
}

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

bool LiteRTParakeetStt::load_vocab(const std::string& path) {
    auto text = json::read_file(path);
    if (text.empty()) return false;

    auto flat = json::parse_flat_object(text);
    for (auto& [key, val] : flat) {
        try {
            int id = std::stoi(key);
            vocab_[id] = val;

            if (val.size() >= 5 && val.size() <= 6 &&
                val.substr(0, 2) == "<|" && val.substr(val.size() - 2) == "|>") {
                lang_tokens_[id] = val.substr(2, val.size() - 4);
            }
        } catch (...) {}
    }

    if (!vocab_.empty()) {
        cfg_.vocab_size   = static_cast<int>(vocab_.size());
        cfg_.blank_id     = cfg_.vocab_size;
        cfg_.total_logits = cfg_.vocab_size + 1 + cfg_.num_dur_bins;
    }

    LOGI("LiteRT Parakeet vocab: %zu tokens, %zu language tokens, blank=%d",
         vocab_.size(), lang_tokens_.size(), cfg_.blank_id);
    return !vocab_.empty();
}

bool LiteRTParakeetStt::set_language(const std::string& language) {
    const std::string code = parakeet::normalize_language_code(language);
    if (code.empty() || code == "auto") {
        clear_language_guidance();
        return true;
    }
    return set_allowed_languages({code});
}

bool LiteRTParakeetStt::set_allowed_languages(const std::vector<std::string>& languages) {
    auto resolved = parakeet::resolve_language_tokens(lang_tokens_, languages);
    if (resolved.empty()) return false;
    guided_lang_tokens_ = std::move(resolved);
    return true;
}

void LiteRTParakeetStt::clear_language_guidance() {
    guided_lang_tokens_.clear();
}

// Decode token ids → text. SentencePiece ▁ (U+2581, UTF-8 E2 96 81) marks a
// word boundary → space. Language tokens are skipped (detected separately).
std::string LiteRTParakeetStt::decode_tokens(const std::vector<int>& token_ids) {
    std::string text;
    for (int id : token_ids) {
        if (lang_tokens_.count(id)) continue;
        auto it = vocab_.find(id);
        if (it == vocab_.end()) continue;

        const std::string& token = it->second;
        if (!token.empty() && static_cast<unsigned char>(token[0]) == 0xE2) {
            if (!text.empty()) text += ' ';
            text += token.substr(3);  // skip the 3-byte ▁
        } else {
            text += token;
        }
    }
    return text;
}

// ---------------------------------------------------------------------------
// Mel spectrogram — NeMo AudioToMelSpectrogramPreprocessor recipe (Parakeet
// TDT v3): pre-emphasis → Slaney-mel STFT (center, log floor 2^-24) →
// per-feature zero-mean unit-variance. Must match the cloud wrapper bit-for-bit
// because the mel feeds the encoder; any drift changes the transcript.
// ---------------------------------------------------------------------------

std::vector<float> LiteRTParakeetStt::compute_mel(const float* audio, size_t length) {
    if (length == 0) return {};

    const float* mel_audio = audio;
    std::vector<float> emphasized;
    if (cfg_.pre_emphasis > 0.0f) {
        emphasized.resize(length);
        emphasized[0] = audio[0];
        for (size_t i = 1; i < length; ++i)
            emphasized[i] = audio[i] - cfg_.pre_emphasis * audio[i - 1];
        mel_audio = emphasized.data();
    }

    constexpr float kLogFloor = 5.960464477539063e-8f;  // 2^-24
    auto mel = audio::mel_spectrogram(
        mel_audio, length,
        cfg_.sample_rate, cfg_.n_fft, cfg_.hop_length, cfg_.win_length,
        cfg_.num_mel_bins,
        /*slaney_norm=*/true, /*log_floor=*/kLogFloor, /*center=*/true);

    const int n_mels = cfg_.num_mel_bins;
    const int frames = (n_mels > 0) ? static_cast<int>(mel.size() / n_mels) : 0;
    if (frames <= 0) return mel;

    constexpr float eps = 1e-5f;
    for (int m = 0; m < n_mels; ++m) {
        float* row = mel.data() + static_cast<size_t>(m) * frames;
        double sum = 0.0;
        for (int t = 0; t < frames; ++t) sum += row[t];
        float mean = static_cast<float>(sum / frames);
        double var = 0.0;
        for (int t = 0; t < frames; ++t) { float d = row[t] - mean; var += d * d; }
        float sd = std::sqrt(static_cast<float>(var / frames) + eps);
        for (int t = 0; t < frames; ++t) row[t] = (row[t] - mean) / sd;
    }
    return mel;
}

// ---------------------------------------------------------------------------
// Encoder (fixed-shape [1, n_mels, enc_mel_frames], chunk + concat) + TDT decode
//
// The soniqo Parakeet-TDT-0.6B-v3-LiteRT-INT8 encoder is FIXED-shape: pad/chunk
// the mel into enc_mel_frames-wide windows, encode each, and concatenate the
// encoder outputs before decoding. Encoder I/O (prod-verified order):
//   in:  [audio_signal float[1,n_mels,T], length int64[1]]
//   out: [length int64[1], encoded float[1, hidden, enc_T]]
// (Note: output order is [length, encoded] — opposite of an earlier
// dynamic-shape export. This matches what the cloud runs in production.)
// ---------------------------------------------------------------------------

LiteRTParakeetStt::DecodeResult LiteRTParakeetStt::decode(const float* audio, size_t length) {
    auto mel = compute_mel(audio, length);
    const int real_frames = static_cast<int>(mel.size() / cfg_.num_mel_bins);
    if (real_frames <= 0) return {};

    auto env = LiteRTEngine::get().env();
    const int kT = cfg_.enc_mel_frames;

    // Query the encoder's static output layouts once: out[1] = encoded
    // [1, hidden, enc_T]. update_allocation=false because the shape is static.
    LiteRtLayout enc_out_layouts[2]{};
    litert_check(LiteRtGetCompiledModelOutputTensorLayouts(
                     enc_compiled_, 0, 2, enc_out_layouts, /*update_allocation=*/false),
                 "GetOutputTensorLayouts(encoder)");
    const int hidden = static_cast<int>(enc_out_layouts[1].dimensions[1]);
    const int enc_T  = static_cast<int>(enc_out_layouts[1].dimensions[2]);
    if (hidden <= 0 || enc_T <= 0) return {};

    auto t_audio   = make_type(kLiteRtElementTypeFloat32, {1, cfg_.num_mel_bins, kT});
    auto t_length  = make_type(kLiteRtElementTypeInt64,   {1});
    auto t_enc_len = make_type(kLiteRtElementTypeInt64,   {1});
    auto t_encoded = make_type(kLiteRtElementTypeFloat32, {1, hidden, enc_T});

    std::vector<float> time_major;  // [total_enc_frames, hidden]
    int total_enc_frames = 0;

    std::vector<float> padded(static_cast<size_t>(cfg_.num_mel_bins) * kT);
    std::vector<float> enc_chunk(static_cast<size_t>(hidden) * enc_T);

    for (int chunk_start = 0; chunk_start < real_frames; chunk_start += kT) {
        const int chunk_frames = std::min(kT, real_frames - chunk_start);

        std::fill(padded.begin(), padded.end(), 0.0f);
        for (int bin = 0; bin < cfg_.num_mel_bins; ++bin) {
            const float* src = mel.data() + static_cast<size_t>(bin) * real_frames + chunk_start;
            std::copy(src, src + chunk_frames,
                      padded.begin() + static_cast<size_t>(bin) * kT);
        }

        int64_t mel_len = chunk_frames;
        int64_t out_len = 0;

        LiteRtHostBuffer in_audio (env, t_audio,   padded.size() * sizeof(float), padded.data());
        LiteRtHostBuffer in_length(env, t_length,  sizeof(int64_t),               &mel_len);
        LiteRtHostBuffer out_lenb (env, t_enc_len, sizeof(int64_t));
        LiteRtHostBuffer out_enc  (env, t_encoded, enc_chunk.size() * sizeof(float));

        LiteRtTensorBuffer enc_ins [2] = { in_audio.raw(), in_length.raw() };
        LiteRtTensorBuffer enc_outs[2] = { out_lenb.raw(),  out_enc.raw()  };
        litert_check(LiteRtRunCompiledModel(enc_compiled_, 0, 2, enc_ins, 2, enc_outs),
                     "Encoder Run");

        out_lenb.read(&out_len,          sizeof(int64_t));
        out_enc .read(enc_chunk.data(),  enc_chunk.size() * sizeof(float));

        // enc_chunk is channels-first [hidden, enc_T]; keep only the valid
        // frames the encoder reported, appended in time-major order.
        const int enc_frames = std::min(static_cast<int>(out_len), enc_T);
        for (int t = 0; t < enc_frames; ++t) {
            for (int ch = 0; ch < hidden; ++ch) {
                time_major.push_back(enc_chunk[static_cast<size_t>(ch) * enc_T + t]);
            }
        }
        total_enc_frames += enc_frames;
    }

    if (total_enc_frames <= 0) return {};

    // Transpose time-major [total, hidden] → channels-first [hidden, total]
    // for tdt_decode (which gathers frame t at stride ch*total + t).
    std::vector<float> encoded_chw(static_cast<size_t>(hidden) * total_enc_frames);
    for (int t = 0; t < total_enc_frames; ++t) {
        for (int ch = 0; ch < hidden; ++ch) {
            encoded_chw[static_cast<size_t>(ch) * total_enc_frames + t] =
                time_major[static_cast<size_t>(t) * hidden + ch];
        }
    }

    LOGI("LiteRT STT: real_frames=%d enc_frames=%d hidden=%d audio=%zu",
         real_frames, total_enc_frames, hidden, length);

    auto result = tdt_decode(encoded_chw.data(), total_enc_frames, hidden);
    LOGI("LiteRT STT: text='%.60s' conf=%.4f", result.text.c_str(), result.confidence);
    return result;
}

// ---------------------------------------------------------------------------
// Public batch transcribe (STTInterface)
// ---------------------------------------------------------------------------

TranscriptionResult LiteRTParakeetStt::transcribe(
    const float* audio, size_t length, int /*sample_rate*/)
{
    auto r = decode(audio, length);
    TranscriptionResult out;
    out.text       = std::move(r.text);
    out.language   = std::move(r.language);
    out.confidence = r.confidence;
    return out;
}

// ---------------------------------------------------------------------------
// TDT greedy decode with the fused decoder-joint model.
//
// Decoder I/O (prod-verified soniqo export):
//   in:  h[layers,1,hidden_dec]  c[layers,1,hidden_dec]  enc_frame[1,1,hidden]
//        target[1,1] int64
//   out: h_new[layers,1,hidden_dec]  logits[1,1,1,total_logits]
//        c_new[layers,1,hidden_dec]
//
// TDT duration bins live at logits[blank+1 ..]. Duration 0 means "emit a token
// and STAY on this frame" — the model may emit up to kMaxSymbols tokens at one
// frame before we force-advance. This dur-0 handling is the fix that the older
// speech-core decode lacked.
// ---------------------------------------------------------------------------

LiteRTParakeetStt::DecodeResult LiteRTParakeetStt::tdt_decode(
    const float* encoded, int64_t enc_len, int64_t hidden)
{
    const int kState = cfg_.decoder_layers * cfg_.decoder_hidden;  // [layers,1,hidden_dec]
    const int kVocab = cfg_.vocab_size;
    const int kBlank = cfg_.blank_id;
    const int kDurs  = cfg_.num_dur_bins;
    constexpr int kMaxSymbols = 10;  // matches NeMo greedy.max_symbols

    auto env      = LiteRTEngine::get().env();
    auto t_state  = make_type(kLiteRtElementTypeFloat32,
                              {cfg_.decoder_layers, 1, cfg_.decoder_hidden});
    auto t_enc    = make_type(kLiteRtElementTypeFloat32, {1, 1, static_cast<int32_t>(hidden)});
    auto t_target = make_type(kLiteRtElementTypeInt64,   {1, 1});
    auto t_logits = make_type(kLiteRtElementTypeFloat32, {1, 1, 1, cfg_.total_logits});

    std::vector<float> h(kState, 0.0f), c(kState, 0.0f);
    std::vector<float> enc_frame(static_cast<size_t>(hidden));
    std::vector<float> logits(static_cast<size_t>(cfg_.total_logits));
    std::vector<float> h_out(kState), c_out(kState);

    std::vector<int> tokens;
    std::string detected_lang;
    float total_log_prob = 0.0f;
    int   total_tokens   = 0;
    int   prev_token     = kBlank;
    int   symbols_at_t   = 0;

    int64_t t = 0;
    while (t < enc_len) {
        for (int64_t ch = 0; ch < hidden; ++ch) {
            enc_frame[ch] = encoded[ch * enc_len + t];
        }
        int64_t target = prev_token;

        LiteRtHostBuffer in_h     (env, t_state,  h.size() * sizeof(float),         h.data());
        LiteRtHostBuffer in_c     (env, t_state,  c.size() * sizeof(float),         c.data());
        LiteRtHostBuffer in_enc   (env, t_enc,    enc_frame.size() * sizeof(float), enc_frame.data());
        LiteRtHostBuffer in_target(env, t_target, sizeof(int64_t),                  &target);
        LiteRtHostBuffer out_h    (env, t_state,  h_out.size() * sizeof(float));
        LiteRtHostBuffer out_log  (env, t_logits, logits.size() * sizeof(float));
        LiteRtHostBuffer out_c    (env, t_state,  c_out.size() * sizeof(float));

        LiteRtTensorBuffer ins [4] = { in_h.raw(), in_c.raw(), in_enc.raw(), in_target.raw() };
        LiteRtTensorBuffer outs[3] = { out_h.raw(), out_log.raw(), out_c.raw() };
        litert_check(LiteRtRunCompiledModel(dec_compiled_, 0, 4, ins, 3, outs),
                     "Decoder Run");

        out_h  .read(h_out.data(),  h_out.size()  * sizeof(float));
        out_log.read(logits.data(), logits.size() * sizeof(float));
        out_c  .read(c_out.data(),  c_out.size()  * sizeof(float));

        // argmax over tokens + blank (indices 0..kVocab inclusive).
        int   best_token = 0;
        float best_logit = logits[0];
        for (int i = 1; i <= kVocab; ++i) {
            if (logits[i] > best_logit) { best_logit = logits[i]; best_token = i; }
        }
        best_token = parakeet::apply_language_guidance(
            logits.data(), best_token, &best_logit, lang_tokens_, guided_lang_tokens_);

        // argmax over the duration bins stored at [kBlank+1 ..].
        int   best_dur       = 0;
        float best_dur_logit = logits[kBlank + 1];
        for (int d = 1; d < kDurs; ++d) {
            float dl = logits[kBlank + 1 + d];
            if (dl > best_dur_logit) { best_dur_logit = dl; best_dur = d; }
        }

        if (best_token != kBlank) {
            tokens.push_back(best_token);
            total_log_prob += best_logit;
            ++total_tokens;
            ++symbols_at_t;
            prev_token = best_token;
            auto lang_it = lang_tokens_.find(best_token);
            if (lang_it != lang_tokens_.end()) detected_lang = lang_it->second;

            // Commit LSTM state ONLY on non-blank emission. The predictor
            // input changes only when we emit a token; on blank we keep
            // the prior state and re-run the joint with the same predictor
            // context against the next encoder frame. The ORT wrapper at
            // src/models/parakeet/parakeet_stt.cpp:348-355 does the same
            // — wiring this differently here was the root cause of the
            // 34.6% LibriSpeech WER (vs 5.7% on ORT) reported in the
            // bench/wer_*.csv comparison; mismatched predictor state on
            // blank cascades produced empty utterances and mid-sentence
            // collapse for ~half of dev-clean.
            h = h_out;
            c = c_out;
        }

        if (best_token == kBlank || best_dur > 0 || symbols_at_t >= kMaxSymbols) {
            t += std::max(best_dur, 1);
            symbols_at_t = 0;
        }
    }

    DecodeResult result;
    result.text     = decode_tokens(tokens);
    result.language = detected_lang;
    result.confidence = total_tokens > 0
        ? std::exp(total_log_prob / static_cast<float>(total_tokens)) : 0.0f;
    if (!result.language.empty()) {
        LOGI("LiteRT STT: detected language=%s", result.language.c_str());
    }
    return result;
}

// ---------------------------------------------------------------------------
// Streaming (accumulate-and-re-transcribe; single-utterance sessions)
// ---------------------------------------------------------------------------

void LiteRTParakeetStt::begin_stream(int sample_rate) {
    stream_buffer_.clear();
    stream_sample_rate_ = sample_rate;
    streaming_ = true;
}

PartialResult LiteRTParakeetStt::push_chunk(const float* audio, size_t length) {
    stream_buffer_.insert(stream_buffer_.end(), audio, audio + length);
    if (stream_buffer_.size() < static_cast<size_t>(stream_sample_rate_ / 2)) return {};

    auto r = decode(stream_buffer_.data(), stream_buffer_.size());
    PartialResult out;
    out.text       = std::move(r.text);
    out.language   = std::move(r.language);
    out.confidence = r.confidence;
    return out;
}

TranscriptionResult LiteRTParakeetStt::end_stream() {
    streaming_ = false;
    if (stream_buffer_.empty()) return {};

    auto r = decode(stream_buffer_.data(), stream_buffer_.size());
    stream_buffer_.clear();

    TranscriptionResult out;
    out.text       = std::move(r.text);
    out.language   = std::move(r.language);
    out.confidence = r.confidence;
    return out;
}

void LiteRTParakeetStt::cancel_stream() {
    stream_buffer_.clear();
    streaming_ = false;
}

void LiteRTParakeetStt::flush_stream() {
    // No-op — single-utterance sessions only.
}

}  // namespace speech_core
