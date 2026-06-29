#pragma once

#include "speech_core/tts_c.h"
#include "speech_core/tts_synthesis_options.h"

#include <stdexcept>
#include <string>

namespace speech_core {

inline bool convert_c_tts_synthesis_options(
    const sc_tts_synthesis_options_t* options,
    TtsSynthesisOptions& out,
    std::string& error,
    const char* owner) {
    out = TtsSynthesisOptions {};
    error.clear();
    if (options == nullptr) {
        return true;
    }

    const std::string owner_name = owner && *owner ? owner : "TTS";
    if (options->struct_size < sizeof(sc_tts_synthesis_options_t)) {
        error = "invalid " + owner_name + " synthesis options struct_size";
        return false;
    }

    switch (options->mode) {
    case SC_TTS_SYNTHESIS_STREAMING:
        out.mode = TtsSynthesisMode::Streaming;
        break;
    case SC_TTS_SYNTHESIS_BUFFERED:
        out.mode = TtsSynthesisMode::Buffered;
        break;
    default:
        error = "unsupported " + owner_name + " synthesis mode";
        return false;
    }

    out.postprocess_flags = options->postprocess_flags;

    try {
        validate_tts_synthesis_options(out, owner_name.c_str());
    } catch (const std::invalid_argument& e) {
        error = e.what();
        return false;
    }

    return true;
}

}  // namespace speech_core
