#ifndef SPEECH_CORE_TTS_C_H
#define SPEECH_CORE_TTS_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum sc_tts_synthesis_mode {
    SC_TTS_SYNTHESIS_STREAMING = 0,
    SC_TTS_SYNTHESIS_BUFFERED = 1,
} sc_tts_synthesis_mode_t;

typedef uint32_t sc_tts_postprocess_flags_t;

enum {
    SC_TTS_POSTPROCESS_NONE = 0u,
    SC_TTS_POSTPROCESS_DEESSER = 1u << 0,
};

typedef struct sc_tts_synthesis_options {
    /// Set to sizeof(sc_tts_synthesis_options_t). Allows future fields.
    uint32_t struct_size;
    sc_tts_synthesis_mode_t mode;
    sc_tts_postprocess_flags_t postprocess_flags;
} sc_tts_synthesis_options_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPEECH_CORE_TTS_C_H
