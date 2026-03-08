#pragma once

// speech-core: Cross-platform voice agent pipeline engine
// https://github.com/soniqo/speech-core

#include "speech_core/interfaces.h"
#include "speech_core/pipeline/voice_pipeline.h"
#include "speech_core/pipeline/turn_detector.h"
#include "speech_core/pipeline/speech_queue.h"
#include "speech_core/pipeline/conversation_context.h"
#include "speech_core/pipeline/agent_config.h"
#include "speech_core/vad/streaming_vad.h"
#include "speech_core/vad/vad_config.h"
#include "speech_core/audio/audio_buffer.h"
#include "speech_core/audio/resampler.h"
#include "speech_core/audio/pcm_codec.h"
#include "speech_core/protocol/realtime_protocol.h"
#include "speech_core/protocol/events.h"
