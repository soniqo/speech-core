#include "speech_core/models/litert_parakeet_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace speech_core {

// ---------------------------------------------------------------------------
// SentencePiece U+2581 → space, then trim
// ---------------------------------------------------------------------------

static void replace_sp_marker(std::string& s) {
    const std::string marker = "\xE2\x96\x81";
    size_t pos = 0;
    while ((pos = s.find(marker, pos)) != std::string::npos) {
        s.replace(pos, marker.size(), " ");
        pos += 1;
    }
}

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

std::string LiteRTParakeetStt::decode_tokens(const std::vector<int>& token_ids) {
    std::string pieces;
    for (int id : token_ids) {
        auto it = vocab_.find(id);
        if (it != vocab_.end()) pieces += it->second;
    }
    replace_sp_marker(pieces);

    size_t start = pieces.find_first_not_of(' ');
    if (start == std::string::npos) return "";
    size_t end = pieces.find_last_not_of(' ');
    return pieces.substr(start, end - start + 1);
}

// ---------------------------------------------------------------------------
// Mel spectrogram (per-utterance mean/var normalised, same as ORT wrapper)
// ---------------------------------------------------------------------------

std::vector<float> LiteRTParakeetStt::compute_mel(const float* audio, size_t length) {
    std::vector<float> emphasized(length);
    emphasized[0] = audio[0];
    for (size_t i = 1; i < length; i++) {
        emphasized[i] = audio[i] - cfg_.pre_emphasis * audio[i - 1];
    }

    auto mel = audio::mel_spectrogram(
        emphasized.data(), emphasized.size(),
        cfg_.sample_rate, cfg_.n_fft, cfg_.hop_length,
        cfg_.win_length, cfg_.num_mel_bins);

    int num_frames = static_cast<int>(mel.size() / cfg_.num_mel_bins);
    if (num_frames > 1) {
        for (int m = 0; m < cfg_.num_mel_bins; m++) {
            float sum = 0, sq_sum = 0;
            for (int t = 0; t < num_frames; t++) {
                float v = mel[m * num_frames + t];
                sum    += v;
                sq_sum += v * v;
            }
            float mean   = sum / num_frames;
            float var    = sq_sum / num_frames - mean * mean;
            float stddev = (var > 0) ? std::sqrt(var) : 1.0f;
            for (int t = 0; t < num_frames; t++) {
                mel[m * num_frames + t] = (mel[m * num_frames + t] - mean) / stddev;
            }
        }
    }
    return mel;
}

// ---------------------------------------------------------------------------
// Encoder (dynamic T) + TDT decode
// ---------------------------------------------------------------------------

LiteRTParakeetStt::DecodeResult LiteRTParakeetStt::decode(const float* audio, size_t length) {
    auto mel = compute_mel(audio, length);
    const int64_t num_frames = static_cast<int64_t>(mel.size() / cfg_.num_mel_bins);

    // Resize the encoder's dynamic time dim to match the current input.
    const int enc_audio_dims[] = {1, cfg_.num_mel_bins, static_cast<int>(num_frames)};
    litert_check(LiteRtCompiledModelResizeInputTensorNonStrict(
                     enc_compiled_, 0, 0, enc_audio_dims, 3),
                 "Encoder ResizeInputTensor(audio_signal)");
    const int enc_len_dims[] = {1};
    litert_check(LiteRtCompiledModelResizeInputTensorNonStrict(
                     enc_compiled_, 0, 1, enc_len_dims, 1),
                 "Encoder ResizeInputTensor(length)");

    // Query the resized output layouts. update_allocation=true propagates the
    // new dynamic shape from the resized inputs to the outputs (encoded[0]
    // shape = [1, hidden, T']; encoded_lengths[1] shape = [1]).
    LiteRtLayout enc_out_layouts[2]{};
    litert_check(LiteRtGetCompiledModelOutputTensorLayouts(
                     enc_compiled_, 0, 2, enc_out_layouts, /*update_allocation=*/true),
                 "GetOutputTensorLayouts(encoder)");
    const int64_t hidden = enc_out_layouts[0].dimensions[1];
    const int64_t enc_t  = enc_out_layouts[0].dimensions[2];

    std::vector<float> encoded(static_cast<size_t>(hidden * enc_t));
    int64_t mel_len = num_frames;
    int64_t enc_len = 0;

    auto env       = LiteRTEngine::get().env();
    auto t_audio   = make_type(kLiteRtElementTypeFloat32,
                                {1, cfg_.num_mel_bins, static_cast<int32_t>(num_frames)});
    auto t_length  = make_type(kLiteRtElementTypeInt64,  {1});
    auto t_encoded = make_type(kLiteRtElementTypeFloat32,
                                {1, static_cast<int32_t>(hidden), static_cast<int32_t>(enc_t)});
    auto t_enc_len = make_type(kLiteRtElementTypeInt64,  {1});

    LiteRtHostBuffer in_audio   (env, t_audio,   mel.size() * sizeof(float), mel.data());
    LiteRtHostBuffer in_length  (env, t_length,  sizeof(int64_t),            &mel_len);
    LiteRtHostBuffer out_encoded(env, t_encoded, encoded.size() * sizeof(float));
    LiteRtHostBuffer out_enc_len(env, t_enc_len, sizeof(int64_t));

    LiteRtTensorBuffer enc_ins[2]  = { in_audio.raw(),    in_length.raw()  };
    LiteRtTensorBuffer enc_outs[2] = { out_encoded.raw(), out_enc_len.raw() };
    litert_check(LiteRtRunCompiledModel(enc_compiled_, 0, 2, enc_ins, 2, enc_outs),
                 "Encoder Run");

    out_encoded.read(encoded.data(), encoded.size() * sizeof(float));
    out_enc_len.read(&enc_len,       sizeof(int64_t));

    LOGI("LiteRT STT: frames=%lld enc_len=%lld hidden=%lld audio=%zu",
         (long long)num_frames, (long long)enc_len, (long long)hidden, length);

    auto result = tdt_decode(encoded.data(), enc_len, hidden);
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
// TDT greedy decoding with fused decoder_joint model
// ---------------------------------------------------------------------------

LiteRTParakeetStt::DecodeResult LiteRTParakeetStt::tdt_decode(
    const float* encoded, int64_t enc_len, int64_t hidden)
{
    std::vector<int> token_ids;
    std::string detected_language;
    float log_prob_sum = 0.0f;
    int   log_prob_count = 0;

    const int64_t state_size = cfg_.decoder_layers * 1 * cfg_.decoder_hidden;
    std::vector<float> h_state(state_size, 0.0f);
    std::vector<float> c_state(state_size, 0.0f);

    int64_t prev_token = static_cast<int64_t>(cfg_.blank_id);
    int64_t t = 0;

    std::vector<float> enc_frame(hidden);
    std::vector<float> logits   (cfg_.total_logits);
    std::vector<float> h_out    (state_size);
    std::vector<float> c_out    (state_size);

    // Decoder I/O tensor types are fixed across the loop.
    auto env      = LiteRTEngine::get().env();
    auto t_enc    = make_type(kLiteRtElementTypeFloat32, {1, static_cast<int32_t>(hidden)});
    auto t_target = make_type(kLiteRtElementTypeInt64,   {1});
    auto t_state  = make_type(kLiteRtElementTypeFloat32, {cfg_.decoder_layers, 1, cfg_.decoder_hidden});
    auto t_logits = make_type(kLiteRtElementTypeFloat32, {1, cfg_.total_logits});

    while (t < enc_len) {
        // Encoder frame at time t: [1, 1, hidden] — note LiteRT decoder takes
        // (B, T_enc=1, H) per convert_litert.py, vs ORT's (B, H, T_enc=1).
        for (int64_t h = 0; h < hidden; h++) {
            enc_frame[h] = encoded[h * enc_len + t];
        }

        LiteRtHostBuffer in_enc    (env, t_enc,    enc_frame.size() * sizeof(float), enc_frame.data());
        LiteRtHostBuffer in_target (env, t_target, sizeof(int64_t),                  &prev_token);
        LiteRtHostBuffer in_h      (env, t_state,  h_state.size() * sizeof(float),   h_state.data());
        LiteRtHostBuffer in_c      (env, t_state,  c_state.size() * sizeof(float),   c_state.data());
        LiteRtHostBuffer out_logits(env, t_logits, logits.size() * sizeof(float));
        LiteRtHostBuffer out_h     (env, t_state,  h_out.size() * sizeof(float));
        LiteRtHostBuffer out_c     (env, t_state,  c_out.size() * sizeof(float));

        LiteRtTensorBuffer ins[4]  = { in_enc.raw(), in_target.raw(), in_h.raw(), in_c.raw() };
        LiteRtTensorBuffer outs[3] = { out_logits.raw(), out_h.raw(), out_c.raw() };
        litert_check(LiteRtRunCompiledModel(dec_compiled_, 0, 4, ins, 3, outs),
                     "Decoder Run");
        out_logits.read(logits.data(), logits.size() * sizeof(float));
        out_h     .read(h_out.data(),  h_out.size()  * sizeof(float));
        out_c     .read(c_out.data(),  c_out.size()  * sizeof(float));

        const int token_end = cfg_.vocab_size + 1;
        int   best_token = 0;
        float best_score = logits[0];
        for (int i = 1; i < token_end; i++) {
            if (logits[i] > best_score) {
                best_score = logits[i];
                best_token = i;
            }
        }

        if (best_token == cfg_.blank_id) {
            t += 1;
        } else {
            if (best_token >= cfg_.first_text_token && best_token < cfg_.vocab_size) {
                auto lang_it = lang_tokens_.find(best_token);
                if (lang_it != lang_tokens_.end()) {
                    if (detected_language.empty()) detected_language = lang_it->second;
                } else {
                    token_ids.push_back(best_token);
                    log_prob_sum += best_score;
                    log_prob_count++;
                }
            }

            const float* dur_logits = logits.data() + token_end;
            int   dur_idx = 0;
            float best_dur = dur_logits[0];
            for (int d = 1; d < cfg_.num_dur_bins; d++) {
                if (dur_logits[d] > best_dur) {
                    best_dur = dur_logits[d];
                    dur_idx  = d;
                }
            }
            t += std::max(cfg_.duration_bins[dur_idx], 1);

            prev_token = best_token;
            h_state = h_out;
            c_state = c_out;
        }
    }

    DecodeResult result;
    result.text     = decode_tokens(token_ids);
    result.language = detected_language;
    if (log_prob_count > 0) {
        float mean_logit  = log_prob_sum / static_cast<float>(log_prob_count);
        result.confidence = 1.0f / (1.0f + std::exp(-mean_logit * 0.1f));
    }
    if (!result.language.empty()) {
        LOGI("LiteRT STT: detected language=%s", result.language.c_str());
    }
    return result;
}

// ---------------------------------------------------------------------------
// Streaming (same accumulate-and-re-transcribe shape as ORT)
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
