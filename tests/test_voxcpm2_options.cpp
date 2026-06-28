// Force-enable asserts even under Release builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/interfaces.h"
#include "speech_core/models/voxcpm2_synthesis_options.h"
#include "speech_core/tts_c.h"
#include "speech_core/tts_synthesis_options.h"
#include "speech_core/voxcpm2_c.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace speech_core;

void test_cpp_defaults() {
    const TtsSynthesisOptions generic_options;
    const VoxCPM2SynthesisOptions options;

    assert(generic_options.mode == TtsSynthesisMode::Streaming);
    assert(generic_options.postprocess_flags == kTtsPostProcessNone);
    assert(options.mode == VoxCPM2SynthesisMode::Streaming);
    assert(options.postprocess_flags == kVoxCPM2PostProcessNone);
    std::printf("  PASS: cpp_defaults\n");
}

void test_voxcpm2_options_alias_generic_tts_options() {
    assert(static_cast<std::uint32_t>(TtsSynthesisMode::Streaming)
           == static_cast<std::uint32_t>(VoxCPM2SynthesisMode::Streaming));
    assert(static_cast<std::uint32_t>(TtsSynthesisMode::Buffered)
           == static_cast<std::uint32_t>(VoxCPM2SynthesisMode::Buffered));
    assert(kTtsPostProcessNone == kVoxCPM2PostProcessNone);
    assert(kTtsPostProcessDeEsser == kVoxCPM2PostProcessDeEsser);
    assert(SC_TTS_SYNTHESIS_STREAMING == SC_VOXCPM2_SYNTHESIS_STREAMING);
    assert(SC_TTS_SYNTHESIS_BUFFERED == SC_VOXCPM2_SYNTHESIS_BUFFERED);
    assert(SC_TTS_POSTPROCESS_NONE == SC_VOXCPM2_POSTPROCESS_NONE);
    assert(SC_TTS_POSTPROCESS_DEESSER == SC_VOXCPM2_POSTPROCESS_DEESSER);
    std::printf("  PASS: voxcpm2_options_alias_generic_tts_options\n");
}

void test_c_abi_values_match_cpp() {
    assert(static_cast<std::uint32_t>(VoxCPM2SynthesisMode::Streaming)
           == static_cast<std::uint32_t>(SC_VOXCPM2_SYNTHESIS_STREAMING));
    assert(static_cast<std::uint32_t>(VoxCPM2SynthesisMode::Buffered)
           == static_cast<std::uint32_t>(SC_VOXCPM2_SYNTHESIS_BUFFERED));
    assert(kVoxCPM2PostProcessNone == SC_VOXCPM2_POSTPROCESS_NONE);
    assert(kVoxCPM2PostProcessDeEsser == SC_VOXCPM2_POSTPROCESS_DEESSER);
    std::printf("  PASS: c_abi_values_match_cpp\n");
}

void test_c_options_can_or_flags() {
    sc_voxcpm2_synthesis_options_t options {};
    options.struct_size = sizeof(options);
    options.mode = SC_VOXCPM2_SYNTHESIS_BUFFERED;
    options.postprocess_flags =
        SC_VOXCPM2_POSTPROCESS_DEESSER | SC_VOXCPM2_POSTPROCESS_NONE;

    assert(options.struct_size == sizeof(sc_voxcpm2_synthesis_options_t));
    assert(options.mode == SC_VOXCPM2_SYNTHESIS_BUFFERED);
    assert((options.postprocess_flags & SC_VOXCPM2_POSTPROCESS_DEESSER) != 0);
    std::printf("  PASS: c_options_can_or_flags\n");
}

class ChunkedTts final : public TTSInterface {
public:
    void synthesize(const std::string&,
                    const std::string&,
                    TTSChunkCallback on_chunk) override {
        const float a[] = {1.0f, 2.0f};
        const float b[] = {3.0f};
        on_chunk(a, 2, false);
        on_chunk(b, 1, false);
        on_chunk(nullptr, 0, true);
    }

    int output_sample_rate() const override { return 24000; }
};

void test_base_tts_options_buffer_streaming_chunks() {
    ChunkedTts tts;
    TtsSynthesisOptions options;
    options.mode = TtsSynthesisMode::Buffered;

    int callbacks = 0;
    bool got_final = false;
    std::vector<float> audio;
    tts.synthesize_with_options("hello", "en", options,
        [&](const float* samples, size_t length, bool is_final) {
            ++callbacks;
            got_final = is_final;
            if (samples && length > 0) {
                audio.assign(samples, samples + length);
            }
        });

    assert(callbacks == 1);
    assert(got_final);
    assert((audio == std::vector<float>{1.0f, 2.0f, 3.0f}));
    std::printf("  PASS: base_tts_options_buffer_streaming_chunks\n");
}

void test_streaming_rejects_offline_flags() {
    ChunkedTts tts;
    TtsSynthesisOptions options;
    options.mode = TtsSynthesisMode::Streaming;
    options.postprocess_flags = kTtsPostProcessDeEsser;

    bool threw = false;
    try {
        tts.synthesize_with_options("hello", "en", options,
            [](const float*, size_t, bool) {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::printf("  PASS: streaming_rejects_offline_flags\n");
}

int main() {
    std::printf("test_voxcpm2_options:\n");
    test_cpp_defaults();
    test_voxcpm2_options_alias_generic_tts_options();
    test_c_abi_values_match_cpp();
    test_c_options_can_or_flags();
    test_base_tts_options_buffer_streaming_chunks();
    test_streaming_rejects_offline_flags();
    std::printf("All VoxCPM2 option tests passed.\n");
    return 0;
}
