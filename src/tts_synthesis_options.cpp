#include "speech_core/interfaces.h"

#include "speech_core/audio/offline_spectral_de_esser.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace speech_core {

void validate_tts_synthesis_options(const TtsSynthesisOptions& options,
                                    const char* owner) {
    constexpr TtsPostProcessFlags kSupportedPostprocess = kTtsPostProcessDeEsser;
    const std::string prefix = owner ? std::string(owner) + ": " : std::string();

    if ((options.postprocess_flags & ~kSupportedPostprocess) != 0) {
        throw std::invalid_argument(prefix + "unsupported postprocess flags");
    }

    if (options.mode != TtsSynthesisMode::Streaming
        && options.mode != TtsSynthesisMode::Buffered) {
        throw std::invalid_argument(prefix + "unsupported synthesis mode");
    }

    if (options.mode == TtsSynthesisMode::Streaming
        && options.postprocess_flags != kTtsPostProcessNone) {
        throw std::invalid_argument(
            prefix + "postprocess flags require buffered synthesis mode");
    }
}

std::vector<float> apply_tts_postprocess(const float* samples,
                                         size_t length,
                                         int sample_rate,
                                         TtsPostProcessFlags flags) {
    if (length == 0) {
        return {};
    }
    if (!samples) {
        throw std::invalid_argument("TTS postprocess input is null");
    }

    std::vector<float> processed(samples, samples + length);
    if ((flags & kTtsPostProcessDeEsser) != 0) {
        processed = audio::OfflineSpectralDeEsser::process_mono(
            processed.data(),
            processed.size(),
            sample_rate,
            audio::OfflineSpectralDeEsser::cli_default_parameters());
    }
    return processed;
}

void TTSInterface::synthesize_with_options(const std::string& text,
                                           const std::string& language,
                                           const TtsSynthesisOptions& options,
                                           TTSChunkCallback on_chunk) {
    if (!on_chunk) {
        return;
    }
    validate_tts_synthesis_options(options, "TTS");

    if (options.mode == TtsSynthesisMode::Streaming
        && options.postprocess_flags == kTtsPostProcessNone) {
        synthesize(text, language, std::move(on_chunk));
        return;
    }

    std::vector<float> synthesized_pcm;
    synthesize(text, language,
        [&synthesized_pcm](const float* samples, size_t length, bool /*is_final*/) {
            if (samples && length > 0) {
                synthesized_pcm.insert(
                    synthesized_pcm.end(),
                    samples,
                    samples + length);
            }
        });

    if (!synthesized_pcm.empty()) {
        std::vector<float> processed_pcm = apply_tts_postprocess(
            synthesized_pcm.data(),
            synthesized_pcm.size(),
            output_sample_rate(),
            options.postprocess_flags);
        on_chunk(processed_pcm.data(), processed_pcm.size(), true);
    } else {
        on_chunk(nullptr, 0, true);
    }
}

}  // namespace speech_core
