#pragma once
// Default model-directory resolution for the example CLIs, so the installed
// `speech` package works without a model-dir argument once models are fetched:
//
//   scripts/download_models.sh ~/.cache/speech-core/models
//   speech_transcribe input.wav
//
// ONNX tools:   $SPEECH_MODEL_DIR        else <cache>/models
// VoxCPM2 CLI:  $SPEECH_LITERT_MODEL_DIR else <cache>/soniqo__VoxCPM2-LiteRT
//               (where sc_voxcpm2_create_from_pretrained / hf_fetch place the
//               downloaded bundle — repo id with '/' sanitized to '__')
// <cache>:      $SPEECH_CORE_CACHE_DIR else %LOCALAPPDATA%\speech-core on
//               Windows, else $XDG_CACHE_HOME/speech-core, else
//               ~/.cache/speech-core
//
// Mirrors default_cache_dir() in src/models/litert/voxcpm2_c.cpp — keep the
// two in sync so the CLIs find what the in-library downloader fetched.

#include <cstdlib>
#include <string>

inline std::string speech_example_cache_dir() {
    if (const char* c = std::getenv("SPEECH_CORE_CACHE_DIR"); c && *c) return c;
#ifdef _WIN32
    if (const char* la = std::getenv("LOCALAPPDATA"); la && *la)
        return std::string(la) + "\\speech-core";
#else
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x)
        return std::string(x) + "/speech-core";
    if (const char* h = std::getenv("HOME"); h && *h)
        return std::string(h) + "/.cache/speech-core";
#endif
    return "./speech-core-cache";
}

/// Default directory for the ONNX model set (Silero, Parakeet, Kokoro, …).
inline std::string speech_example_model_dir() {
    if (const char* e = std::getenv("SPEECH_MODEL_DIR"); e && *e) return e;
    return speech_example_cache_dir() + "/models";
}

/// Default directory for the VoxCPM2 LiteRT bundle.
inline std::string speech_example_voxcpm2_dir() {
    if (const char* e = std::getenv("SPEECH_LITERT_MODEL_DIR"); e && *e) return e;
    return speech_example_cache_dir() + "/soniqo__VoxCPM2-LiteRT";
}
