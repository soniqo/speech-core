// Force-enable asserts even under Release builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/models/voxcpm2_synthesis_options.h"
#include "speech_core/voxcpm2_c.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

using namespace speech_core;

void test_cpp_defaults() {
    const VoxCPM2SynthesisOptions options;

    assert(options.mode == VoxCPM2SynthesisMode::Streaming);
    assert(options.postprocess_flags == kVoxCPM2PostProcessNone);
    std::printf("  PASS: cpp_defaults\n");
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

int main() {
    std::printf("test_voxcpm2_options:\n");
    test_cpp_defaults();
    test_c_abi_values_match_cpp();
    test_c_options_can_or_flags();
    std::printf("All VoxCPM2 option tests passed.\n");
    return 0;
}
