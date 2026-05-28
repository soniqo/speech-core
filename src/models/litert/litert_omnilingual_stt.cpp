#include "speech_core/models/litert_omnilingual_stt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace speech_core {

// ---------------------------------------------------------------------------
// SentencePiece tokenizer — minimal protobuf parse (id → piece) for decode.
// ModelProto { repeated SentencePiece pieces = 1; }
// SentencePiece { string piece = 1; float score = 2; Type type = 3; }
// ---------------------------------------------------------------------------

bool LiteRTOmnilingualStt::load_tokenizer(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    f.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(file_size);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(file_size));

    id_to_piece_.clear();
    size_t pos = 0;
    while (pos < file_size) {
        uint8_t tag = buf[pos++];
        int field_num = tag >> 3;
        int wire_type = tag & 0x7;

        if (field_num == 1 && wire_type == 2) {
            uint32_t len = 0;
            int shift = 0;
            while (pos < file_size && (buf[pos] & 0x80)) {
                len |= (buf[pos++] & 0x7f) << shift;
                shift += 7;
            }
            if (pos < file_size) len |= buf[pos++] << shift;

            size_t end = pos + len;
            if (end > file_size) break;

            std::string piece;
            size_t sub = pos;
            while (sub < end) {
                uint8_t stag = buf[sub++];
                int sf = stag >> 3;
                int sw = stag & 0x7;
                if (sf == 1 && sw == 2) {
                    uint32_t slen = 0;
                    int sh = 0;
                    while (sub < end && (buf[sub] & 0x80)) {
                        slen |= (buf[sub++] & 0x7f) << sh;
                        sh += 7;
                    }
                    if (sub < end) slen |= buf[sub++] << sh;
                    if (sub + slen <= end) {
                        piece = std::string(reinterpret_cast<const char*>(&buf[sub]), slen);
                    }
                    sub += slen;
                } else if (sw == 0) {
                    while (sub < end && (buf[sub] & 0x80)) sub++;
                    if (sub < end) sub++;
                } else if (sw == 2) {
                    uint32_t slen = 0;
                    int sh = 0;
                    while (sub < end && (buf[sub] & 0x80)) {
                        slen |= (buf[sub++] & 0x7f) << sh;
                        sh += 7;
                    }
                    if (sub < end) slen |= buf[sub++] << sh;
                    sub += slen;
                } else if (sw == 5) {
                    sub += 4;
                } else if (sw == 1) {
                    sub += 8;
                } else {
                    break;
                }
            }
            id_to_piece_.push_back(piece);
            pos = end;
        } else if (wire_type == 0) {
            while (pos < file_size && (buf[pos] & 0x80)) pos++;
            if (pos < file_size) pos++;
        } else if (wire_type == 2) {
            uint32_t len = 0;
            int shift = 0;
            while (pos < file_size && (buf[pos] & 0x80)) {
                len |= (buf[pos++] & 0x7f) << shift;
                shift += 7;
            }
            if (pos < file_size) len |= buf[pos++] << shift;
            pos += len;
        } else if (wire_type == 5) {
            pos += 4;
        } else if (wire_type == 1) {
            pos += 8;
        } else {
            break;
        }
    }

    LOGI("Omnilingual: loaded %zu tokens from %s", id_to_piece_.size(), path.c_str());
    return !id_to_piece_.empty();
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

LiteRTOmnilingualStt::LiteRTOmnilingualStt(const std::string& model_path,
                                           const std::string& tokenizer_path,
                                           bool hw_accel)
{
    LiteRTEngine::get().load(model_path, hw_accel, &model_, &compiled_);

    if (!load_tokenizer(tokenizer_path)) {
        throw std::runtime_error("Failed to load SentencePiece tokenizer: " + tokenizer_path);
    }

    // Read T (frames per chunk) + vocab from the static output layout
    // [1, T, vocab]. The compiled-model API exposes output layouts only, so
    // derive the input chunk length from T: max_audio = T * (sr / frame_rate).
    LiteRtLayout out_layout[1]{};
    litert_check(LiteRtGetCompiledModelOutputTensorLayouts(
                     compiled_, 0, 1, out_layout, /*update_allocation=*/false),
                 "Omnilingual GetOutputTensorLayouts");
    if (out_layout[0].rank >= 3) {
        frames_per_chunk_ = static_cast<int>(out_layout[0].dimensions[1]);
        cfg_.vocab_size   = static_cast<int>(out_layout[0].dimensions[2]);
    }
    const int down = cfg_.sample_rate / cfg_.frame_rate;  // 320
    if (frames_per_chunk_ > 0) {
        cfg_.max_audio_samples = frames_per_chunk_ * down;
    } else {
        frames_per_chunk_ = cfg_.max_audio_samples / down;
    }

    LOGI("Omnilingual: input=[1,%d] output=[1,%d,%d] (%.1fs chunks)",
         cfg_.max_audio_samples, frames_per_chunk_, cfg_.vocab_size,
         static_cast<float>(cfg_.max_audio_samples) / cfg_.sample_rate);
}

LiteRTOmnilingualStt::~LiteRTOmnilingualStt() {
    if (compiled_) LiteRtDestroyCompiledModel(compiled_);
    if (model_)    LiteRtDestroyModel(model_);
}

// ---------------------------------------------------------------------------
// Transcribe — chunk to max_audio_samples, z-score normalise, CTC decode.
// ---------------------------------------------------------------------------

TranscriptionResult LiteRTOmnilingualStt::transcribe(
    const float* audio, size_t length, int sample_rate)
{
    if (sample_rate != cfg_.sample_rate) {
        throw std::runtime_error("OmnilingualSTT expects " +
                                 std::to_string(cfg_.sample_rate) + " Hz");
    }

    const int chunk_samples = cfg_.max_audio_samples;
    const int down          = cfg_.sample_rate / cfg_.frame_rate;  // 320

    auto env      = LiteRTEngine::get().env();
    auto t_audio  = make_type(kLiteRtElementTypeFloat32, {1, chunk_samples});
    auto t_logits = make_type(kLiteRtElementTypeFloat32, {1, frames_per_chunk_, cfg_.vocab_size});

    std::vector<float> logits(static_cast<size_t>(frames_per_chunk_) * cfg_.vocab_size);
    std::string full_text;

    for (size_t offset = 0; offset < length; offset += chunk_samples) {
        size_t chunk_len = std::min(static_cast<size_t>(chunk_samples), length - offset);

        std::vector<float> normalized(static_cast<size_t>(chunk_samples), 0.0f);
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < chunk_len; ++i) {
            float v = audio[offset + i];
            sum += v;
            sum_sq += static_cast<double>(v) * v;
        }
        float mean    = static_cast<float>(sum / chunk_len);
        float var     = static_cast<float>(sum_sq / chunk_len - static_cast<double>(mean) * mean);
        float std_dev = std::sqrt(std::max(var, 1e-10f));
        for (size_t i = 0; i < chunk_len; ++i) {
            normalized[i] = (audio[offset + i] - mean) / std_dev;
        }

        LiteRtHostBuffer in_audio (env, t_audio,  static_cast<size_t>(chunk_samples) * sizeof(float), normalized.data());
        LiteRtHostBuffer out_log  (env, t_logits, logits.size() * sizeof(float));

        LiteRtTensorBuffer ins [1] = { in_audio.raw() };
        LiteRtTensorBuffer outs[1] = { out_log.raw() };
        litert_check(LiteRtRunCompiledModel(compiled_, 0, 1, ins, 1, outs), "Omnilingual Run");
        out_log.read(logits.data(), logits.size() * sizeof(float));

        int num_frames = static_cast<int>(chunk_len) / down;
        if (num_frames <= 0) continue;

        std::string chunk_text = ctc_decode(logits.data(), num_frames);
        if (!chunk_text.empty()) {
            if (!full_text.empty()) full_text += ' ';
            full_text += chunk_text;
        }
    }

    TranscriptionResult result;
    result.text = std::move(full_text);
    return result;
}

// ---------------------------------------------------------------------------
// CTC greedy decode
// ---------------------------------------------------------------------------

std::string LiteRTOmnilingualStt::ctc_decode(const float* logits, int num_frames) {
    const int V     = cfg_.vocab_size;
    const int blank = cfg_.blank_id;

    std::vector<int> tokens;
    int prev_token = -1;

    for (int t = 0; t < num_frames; ++t) {
        const float* frame = logits + static_cast<size_t>(t) * V;
        int   best     = 0;
        float best_val = frame[0];
        for (int v = 1; v < V; ++v) {
            if (frame[v] > best_val) { best_val = frame[v]; best = v; }
        }
        if (best != blank && best != prev_token) tokens.push_back(best);
        prev_token = best;
    }

    std::string text;
    for (int id : tokens) {
        if (id < 0 || id >= static_cast<int>(id_to_piece_.size())) continue;
        const std::string& piece = id_to_piece_[id];
        if (piece.size() >= 3 &&
            static_cast<unsigned char>(piece[0]) == 0xE2 &&
            static_cast<unsigned char>(piece[1]) == 0x96 &&
            static_cast<unsigned char>(piece[2]) == 0x81) {
            if (!text.empty()) text += ' ';
            text += piece.substr(3);
        } else {
            text += piece;
        }
    }
    return text;
}

}  // namespace speech_core
