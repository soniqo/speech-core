#include "speech_core/models/litert_voxcpm2_tts.h"

#include "speech_core/models/litert_engine.h"

#include <stdexcept>

namespace speech_core {

LiteRTVoxCPM2Tts::LiteRTVoxCPM2Tts(const std::string& text_prefill_path,
                                    const std::string& token_step_path,
                                    const std::string& audio_encoder_path,
                                    const std::string& audio_decoder_path,
                                    const std::string& tokenizer_path,
                                    bool hw_accel)
{
    // The four graphs are independent .tflite files; load each via the shared
    // LiteRT engine. The model handles are owned by this wrapper and freed in
    // the dtor below (interpreter first, then model — required by the C API).
    auto& engine = LiteRTEngine::get();
    text_prefill_  = engine.load(text_prefill_path,  hw_accel, &text_prefill_model_);
    token_step_    = engine.load(token_step_path,    hw_accel, &token_step_model_);
    audio_encoder_ = engine.load(audio_encoder_path, hw_accel, &audio_encoder_model_);
    audio_decoder_ = engine.load(audio_decoder_path, hw_accel, &audio_decoder_model_);

    // Tokenizer is parsed at construction so a malformed tokenizer.json is
    // caught immediately rather than on the first synthesize() call.
    tokenizer_ = std::make_unique<VoxCPM2Tokenizer>(tokenizer_path);
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

void LiteRTVoxCPM2Tts::synthesize(const std::string& /*text*/,
                                   const std::string& /*language*/,
                                   TTSChunkCallback /*on_chunk*/)
{
    // Skeleton: the full pipeline is text_prefill → token_step (×N) → audio_decode,
    // with a HF BPE tokenizer for input and explicit K/V cache passed through
    // every token_step call. Not yet implemented — this wrapper currently only
    // proves the four graphs and the tokenizer file load cleanly.
    throw std::runtime_error(
        "LiteRTVoxCPM2Tts::synthesize is not implemented in the skeleton; "
        "the wrapper currently only verifies the four LiteRT graphs and "
        "tokenizer.json can be loaded.");
}

void LiteRTVoxCPM2Tts::cancel() {
    cancelled_ = true;
}

}  // namespace speech_core
