// Independent HF/torch probes cover centred windows, constant head padding,
// and the last real sample followed by zeros after pre-emphasis. Streaming
// and reset checks also keep the corrected path independent of push sizes.
#include "speech_core/audio/mel.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::printf("FAIL: %s\n", message); ++failures; }
}

struct Probe { size_t samples; size_t frame; size_t band; float value; };
const Probe probes[] = {
#include "data/nemotron_mel_reference.inc"
};

std::vector<float> signal(size_t length) {
    std::vector<float> audio(length);
    std::uint32_t state = 0x12345678u;
    for (size_t i = 0; i < length; ++i) {
        state = state * 1664525u + 1013904223u;
        audio[i] = (static_cast<float>(state >> 8) / (1u << 24) - .5f) / 32.f;
        if (i >= length / 3 && i < std::min(length - 1, length / 3 + 512)) audio[i] = 0.f;
    }
    audio.front() = .03125f;
    audio.back() = .125f;  // exposes a false pre-emphasis pulse from raw zero padding
    return audio;
}

speech_core::audio::StreamingMelSpectrogram::Config config() {
    speech_core::audio::StreamingMelSpectrogram::Config c;
    c.slaney_norm = true;
    c.log_floor = 1.f / (1 << 24);
    c.pre_emphasis = .97f;
    c.torch_stft_layout = true;
    c.center_pad_zeros = true;
    return c;
}
}

int main() {
    float worst_probe = 0.f;
    for (size_t length : {37u, 5120u, 5217u}) {
        const auto audio = signal(length);
        const size_t total_frames = ((length + 5119) / 5120) * 32;
        std::vector<float> baseline;
        for (size_t chunk : {1u, 7u, 97u, 320u, 5120u, 1u << 20}) {
            speech_core::audio::StreamingMelSpectrogram stream(config());
            std::vector<float> frames;
            for (size_t i = 0; i < length; i += chunk) {
                stream.push(audio.data() + i, std::min(chunk, length - i), frames);
                expect(stream.buffered_samples() <= 512 + 400 + 160 + chunk,
                       "audio retained for future frames stays bounded");
            }
            stream.finish(total_frames, frames);
            expect(frames.size() == total_frames * 128, "finish completes every final encoder frame");
            if (frames.size() != total_frames * 128) continue;
            expect(stream.samples_pushed() == length, "padding does not invent captured samples");
            expect(stream.buffered_samples() == 0, "finish releases buffered audio");
            if (baseline.empty()) baseline = frames;
            expect(frames == baseline, "20 ms, tiny and whole-buffer pushes agree");
            for (const auto& probe : probes) {
                if (probe.samples != length) continue;
                const float difference = std::fabs(frames[probe.frame * 128 + probe.band] - probe.value);
                worst_probe = std::max(worst_probe, difference);
                if (difference > 5e-4f) {
                    std::printf("samples=%zu frame=%zu band=%zu expected=%g got=%g\n",
                        length, probe.frame, probe.band, probe.value, frames[probe.frame * 128 + probe.band]);
                    expect(false, "streamed features match the independent HF/torch probes");
                }
            }
            expect(stream.finish(total_frames, frames) == 0, "a second finish emits nothing");
            bool rejected = false;
            try { stream.push(audio.data(), 1, frames); } catch (const std::logic_error&) { rejected = true; }
            expect(rejected, "new audio after finish requires reset");
            stream.reset();
            std::vector<float> repeated;
            stream.push(audio.data(), audio.size(), repeated);
            stream.finish(total_frames, repeated);
            expect(repeated == frames, "reset clears finish and pre-emphasis history");
        }
    }
    std::printf("Nemotron mel: worst reference probe difference=%g; failures=%d\n", worst_probe, failures);
    return failures ? 1 : 0;
}
