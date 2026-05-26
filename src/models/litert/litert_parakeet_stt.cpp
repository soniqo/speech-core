#include "speech_core/models/litert_parakeet_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/models/litert_engine.h"
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

LiteRTParakeetStt::LiteRTParakeetStt(
    const std::string& encoder_path,
    const std::string& decoder_joint_path,
    const std::string& vocab_path,
    bool hw_accel)
{
    auto& engine = LiteRTEngine::get();
    enc_interp_ = engine.load(encoder_path,       hw_accel, &enc_model_);
    dec_interp_ = engine.load(decoder_joint_path, false,    &dec_model_);

    load_vocab(vocab_path);
}

LiteRTParakeetStt::~LiteRTParakeetStt() {
    if (dec_interp_) TfLiteInterpreterDelete(dec_interp_);
    if (dec_model_)  TfLiteModelDelete(dec_model_);
    if (enc_interp_) TfLiteInterpreterDelete(enc_interp_);
    if (enc_model_)  TfLiteModelDelete(enc_model_);
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
                std::string code = val.substr(2, val.size() - 4);
                lang_tokens_[id] = code;
            }
        } catch (...) {}
    }

    if (!vocab_.empty()) {
        cfg_.vocab_size = static_cast<int>(vocab_.size());
        cfg_.blank_id = cfg_.vocab_size;
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
// Mel spectrogram (same as ORT wrapper)
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
                sum += v;
                sq_sum += v * v;
            }
            float mean = sum / num_frames;
            float var = sq_sum / num_frames - mean * mean;
            float stddev = (var > 0) ? std::sqrt(var) : 1.0f;
            for (int t = 0; t < num_frames; t++) {
                mel[m * num_frames + t] = (mel[m * num_frames + t] - mean) / stddev;
            }
        }
    }

    return mel;
}

// ---------------------------------------------------------------------------
// Decode (mel → encoder → tdt_decode)
// ---------------------------------------------------------------------------

LiteRTParakeetStt::DecodeResult LiteRTParakeetStt::decode(const float* audio, size_t length) {
    auto mel = compute_mel(audio, length);
    int64_t num_frames = static_cast<int64_t>(mel.size() / cfg_.num_mel_bins);

    // Encoder input 0: audio_signal [1, 128, T]
    // Encoder input 1: length [1] int64
    const int enc_audio_dims[] = {1, cfg_.num_mel_bins, static_cast<int>(num_frames)};
    litert_check(TfLiteInterpreterResizeInputTensor(enc_interp_, 0, enc_audio_dims, 3),
                 "Encoder ResizeInputTensor(audio_signal)");
    const int enc_len_dims[] = {1};
    litert_check(TfLiteInterpreterResizeInputTensor(enc_interp_, 1, enc_len_dims, 1),
                 "Encoder ResizeInputTensor(length)");
    litert_check(TfLiteInterpreterAllocateTensors(enc_interp_), "Encoder AllocateTensors");

    TfLiteTensor* in_audio  = TfLiteInterpreterGetInputTensor(enc_interp_, 0);
    TfLiteTensor* in_length = TfLiteInterpreterGetInputTensor(enc_interp_, 1);

    litert_check(TfLiteTensorCopyFromBuffer(in_audio, mel.data(),
                                            mel.size() * sizeof(float)),
                 "Encoder CopyFromBuffer(audio_signal)");
    int64_t mel_len = num_frames;
    litert_check(TfLiteTensorCopyFromBuffer(in_length, &mel_len, sizeof(int64_t)),
                 "Encoder CopyFromBuffer(length)");

    litert_check(TfLiteInterpreterInvoke(enc_interp_), "Encoder Invoke");

    // Encoder output 0: encoded [1, hidden, T']
    // Encoder output 1: encoded_lengths [1] int64
    const TfLiteTensor* out_encoded     = TfLiteInterpreterGetOutputTensor(enc_interp_, 0);
    const TfLiteTensor* out_encoded_len = TfLiteInterpreterGetOutputTensor(enc_interp_, 1);

    int32_t enc_dims = TfLiteTensorNumDims(out_encoded);
    int64_t hidden  = (enc_dims >= 3) ? TfLiteTensorDim(out_encoded, 1) : cfg_.encoder_hidden;
    int64_t enc_t   = (enc_dims >= 3) ? TfLiteTensorDim(out_encoded, 2) : 0;

    std::vector<float> encoded(static_cast<size_t>(hidden * enc_t));
    litert_check(TfLiteTensorCopyToBuffer(out_encoded, encoded.data(),
                                          encoded.size() * sizeof(float)),
                 "Encoder CopyToBuffer(encoded)");

    int64_t enc_len = 0;
    litert_check(TfLiteTensorCopyToBuffer(out_encoded_len, &enc_len, sizeof(int64_t)),
                 "Encoder CopyToBuffer(encoded_lengths)");

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
    out.text = std::move(r.text);
    out.language = std::move(r.language);
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
    int log_prob_count = 0;

    int64_t state_size = cfg_.decoder_layers * 1 * cfg_.decoder_hidden;
    std::vector<float> h_state(state_size, 0.0f);
    std::vector<float> c_state(state_size, 0.0f);

    int64_t prev_token = static_cast<int64_t>(cfg_.blank_id);
    int64_t t = 0;

    std::vector<float> enc_frame(hidden);
    std::vector<float> logits(cfg_.total_logits);

    while (t < enc_len) {
        // Encoder frame at time t: [1, 1, hidden] — note LiteRT decoder takes
        // (B, T_enc=1, H) per convert_litert.py, vs ORT's (B, H, T_enc=1).
        for (int64_t h = 0; h < hidden; h++) {
            enc_frame[h] = encoded[h * enc_len + t];
        }

        TfLiteTensor* in_enc    = TfLiteInterpreterGetInputTensor(dec_interp_, 0);
        TfLiteTensor* in_target = TfLiteInterpreterGetInputTensor(dec_interp_, 1);
        TfLiteTensor* in_h      = TfLiteInterpreterGetInputTensor(dec_interp_, 2);
        TfLiteTensor* in_c      = TfLiteInterpreterGetInputTensor(dec_interp_, 3);

        litert_check(TfLiteTensorCopyFromBuffer(in_enc, enc_frame.data(),
                                                enc_frame.size() * sizeof(float)),
                     "Decoder CopyFromBuffer(encoder_out)");
        litert_check(TfLiteTensorCopyFromBuffer(in_target, &prev_token, sizeof(int64_t)),
                     "Decoder CopyFromBuffer(target)");
        litert_check(TfLiteTensorCopyFromBuffer(in_h, h_state.data(),
                                                h_state.size() * sizeof(float)),
                     "Decoder CopyFromBuffer(h)");
        litert_check(TfLiteTensorCopyFromBuffer(in_c, c_state.data(),
                                                c_state.size() * sizeof(float)),
                     "Decoder CopyFromBuffer(c)");

        litert_check(TfLiteInterpreterInvoke(dec_interp_), "Decoder Invoke");

        const TfLiteTensor* out_logits = TfLiteInterpreterGetOutputTensor(dec_interp_, 0);
        const TfLiteTensor* out_h      = TfLiteInterpreterGetOutputTensor(dec_interp_, 1);
        const TfLiteTensor* out_c      = TfLiteInterpreterGetOutputTensor(dec_interp_, 2);

        litert_check(TfLiteTensorCopyToBuffer(out_logits, logits.data(),
                                              logits.size() * sizeof(float)),
                     "Decoder CopyToBuffer(logits)");

        int token_end = cfg_.vocab_size + 1;

        int best_token = 0;
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
                    if (detected_language.empty()) {
                        detected_language = lang_it->second;
                    }
                } else {
                    token_ids.push_back(best_token);
                    log_prob_sum += best_score;
                    log_prob_count++;
                }
            }

            const float* dur_logits = logits.data() + token_end;
            int dur_idx = 0;
            float best_dur = dur_logits[0];
            for (int d = 1; d < cfg_.num_dur_bins; d++) {
                if (dur_logits[d] > best_dur) {
                    best_dur = dur_logits[d];
                    dur_idx = d;
                }
            }
            t += std::max(cfg_.duration_bins[dur_idx], 1);

            prev_token = best_token;

            litert_check(TfLiteTensorCopyToBuffer(out_h, h_state.data(),
                                                  h_state.size() * sizeof(float)),
                         "Decoder CopyToBuffer(h_new)");
            litert_check(TfLiteTensorCopyToBuffer(out_c, c_state.data(),
                                                  c_state.size() * sizeof(float)),
                         "Decoder CopyToBuffer(c_new)");
        }
    }

    DecodeResult result;
    result.text = decode_tokens(token_ids);
    result.language = detected_language;

    if (log_prob_count > 0) {
        float mean_logit = log_prob_sum / static_cast<float>(log_prob_count);
        result.confidence = 1.0f / (1.0f + std::exp(-mean_logit * 0.1f));
    }

    if (!result.language.empty()) {
        LOGI("LiteRT STT: detected language=%s", result.language.c_str());
    }

    return result;
}

// ---------------------------------------------------------------------------
// Streaming: accumulate audio and re-transcribe (same shape as ORT wrapper)
// ---------------------------------------------------------------------------

void LiteRTParakeetStt::begin_stream(int sample_rate) {
    stream_buffer_.clear();
    stream_sample_rate_ = sample_rate;
    streaming_ = true;
}

PartialResult LiteRTParakeetStt::push_chunk(const float* audio, size_t length) {
    stream_buffer_.insert(stream_buffer_.end(), audio, audio + length);

    if (stream_buffer_.size() < static_cast<size_t>(stream_sample_rate_ / 2)) {
        return {};
    }

    auto r = decode(stream_buffer_.data(), stream_buffer_.size());
    PartialResult out;
    out.text = std::move(r.text);
    out.language = std::move(r.language);
    out.confidence = r.confidence;
    return out;
}

TranscriptionResult LiteRTParakeetStt::end_stream() {
    streaming_ = false;
    if (stream_buffer_.empty()) return {};

    auto r = decode(stream_buffer_.data(), stream_buffer_.size());
    stream_buffer_.clear();

    TranscriptionResult out;
    out.text = std::move(r.text);
    out.language = std::move(r.language);
    out.confidence = r.confidence;
    return out;
}

void LiteRTParakeetStt::cancel_stream() {
    stream_buffer_.clear();
    streaming_ = false;
}

void LiteRTParakeetStt::flush_stream() {
    // No-op — single-utterance sessions only
}

}  // namespace speech_core
