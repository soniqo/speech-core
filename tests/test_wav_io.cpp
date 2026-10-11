// Keep checks active in Release and sanitizer builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "speech_core/audio/wav_io.h"
#include "wav_test_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <sstream>

using speech_core::WavData;

namespace {

std::string read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    assert(file);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void check_decode(const std::filesystem::path& path, uint16_t format, uint16_t bits,
                  uint16_t channels, const std::vector<uint32_t>& samples,
                  const std::vector<float>& expected) {
    // The odd-sized JUNK chunk also makes float payloads only two-byte aligned.
    wav_test::write_fixture(path, format, bits, channels, 48000, samples, true);
    WavData wav;
    assert(wav.load_mono(path));
    assert(wav.sample_rate == 48000);
    assert(wav.samples.size() == expected.size());
    assert(std::fabs(wav.duration() - static_cast<double>(expected.size()) / 48000) < 1e-12);
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::fabs(wav.samples[i] - expected[i]) < 1e-7f);
    }

    std::istringstream stream(read_bytes(path), std::ios::in | std::ios::binary);
    WavData from_stream;
    assert(from_stream.load_mono(stream));
    assert(from_stream.sample_rate == wav.sample_rate);
    assert(from_stream.samples == wav.samples);
}

void test_formats_and_downmix(const std::filesystem::path& path) {
    check_decode(path, 1, 16, 1, {0x8000, 0xc000, 0, 0x4000, 0x7fff},
                 {-1, -0.5f, 0, 0.5f, 32767.0f / 32768});
    check_decode(path, 1, 24, 1, {0x800000, 0xc00000, 0, 0x400000, 0x7fffff},
                 {-1, -0.5f, 0, 0.5f, 8388607.0f / 8388608});
    check_decode(path, 1, 32, 1, {0x80000000, 0xc0000000, 0, 0x40000000, 0x7fffffff},
                 {-1, -0.5f, 0, 0.5f, 1});
    check_decode(path, 3, 32, 1, {0xbf800000, 0xbf000000, 0, 0x3f000000, 0x3f800000},
                 {-1, -0.5f, 0, 0.5f, 1});
    check_decode(path, 1, 16, 2, {0x8000, 0x4000, 0x4000, 0}, {-0.25f, 0.25f});
    check_decode(path, 1, 24, 2, {0x800000, 0x400000, 0x400000, 0}, {-0.25f, 0.25f});
    check_decode(path, 1, 32, 2, {0x80000000, 0x40000000, 0x40000000, 0}, {-0.25f, 0.25f});
    check_decode(path, 3, 32, 2, {0xbf800000, 0x3f000000, 0x3f000000, 0}, {-0.25f, 0.25f});
}

void test_utf8_write_and_read(const std::filesystem::path& directory) {
    const auto path = directory / std::filesystem::u8path(u8"\u092e\u0947\u0930\u093e-\u00e4udio.wav");
    const WavData input{{-2, -0.5f, 0, 0.5f, 2}, 24000};
    assert(input.save(path));
    WavData output;
    assert(output.load_mono(path));
    assert(output.sample_rate == input.sample_rate);
    assert(output.samples.size() == input.samples.size());
    for (size_t i = 0; i < input.samples.size(); ++i) {
        const float clipped = std::clamp(input.samples[i], -1.0f, 1.0f);
        assert(std::fabs(output.samples[i] - clipped) < 2.0f / 32768);
    }
    const auto missing_parent = directory / "missing" / "out.wav";
    assert(!input.save(missing_parent));
}

void test_invalid_input(const std::filesystem::path& path) {
    WavData wav{{1}, 16000};
    assert(!wav.load_mono((path.u8string() + ".missing")));

    wav_test::write_fixture(path, 1, 8, 1, 16000, {128});
    assert(!wav.load_mono(path));

    wav_test::write_fixture(path, 1, 16, 1, 0, {0});
    assert(!wav.load_mono(path));
    
    wav_test::write_fixture(path, 1, 16, 2, 16000, {0, 0, 0});
    assert(!wav.load_mono(path));

    wav_test::write_fixture(path, 1, 16, 1, 16000, {0, 1});
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 2);
    assert(!wav.load_mono(path));
}

void test_path_overloads_and_legacy_api(const std::filesystem::path& directory) {
    const auto path = directory / std::filesystem::u8path(u8"legacy-\u092e\u0947\u0930\u093e.wav");
    const auto utf8_path = path.u8string();
    const WavData input{{-2, -0.5f, 0, 0.5f, 2}, 24000};
    // Independently encoded fixture pins the existing PCM16 clipping and rounding.
    wav_test::write_fixture(path, 1, 16, 1, 24000, {0x8001, 0xc001, 0, 0x3fff, 0x7fff});
    const auto expected = read_bytes(path);
    auto check_write = [&](bool wrote) {
        assert(wrote);
        assert(read_bytes(path) == expected);
    };
    check_write(input.save(utf8_path));
    check_write(input.save(utf8_path.c_str()));
    check_write(input.save(path));
    check_write(WavData::write_mono(utf8_path, input.samples, input.sample_rate));
    check_write(speech_core::write_wav_mono_pcm16(utf8_path, input.samples, input.sample_rate));
    check_write(speech_core::write_wav_mono_pcm16(
        utf8_path, input.samples.data(), input.samples.size(), input.sample_rate));

    WavData output;
    assert(output.load_mono(utf8_path.c_str()));
    const auto decoded = output.samples;
    assert(output.load_mono(utf8_path));
    assert(output.samples == decoded && output.sample_rate == input.sample_rate);
    assert(output.load(utf8_path));
    assert(output.samples == decoded && output.sample_rate == input.sample_rate);
    assert(speech_core::load_wav_mono_pcm16(utf8_path, &output));
    assert(output.samples == decoded && output.sample_rate == input.sample_rate);
    assert(!speech_core::load_wav_mono_pcm16(utf8_path, nullptr));
    assert(!output.load(utf8_path + ".missing"));
    output = input;
    assert(!speech_core::load_wav_mono_pcm16(utf8_path + ".missing", &output));

    // Literal calls must remain unambiguous alongside string and path overloads.
    assert(!WavData{}.save("unused-invalid-output.wav"));
    assert(!output.load_mono("unused-missing-input.wav"));
    output = input;
    assert(!output.load_mono(static_cast<const char*>(nullptr)));
    assert(!input.save(static_cast<const char*>(nullptr)));
}

template <typename Write>
void check_rejected_write(const std::filesystem::path& directory, Write write) {
    const auto existing = directory / "existing.wav";
    const auto missing = directory / "not-created.wav";
    wav_test::write_fixture(existing, 1, 16, 1, 16000, {0, 0x4000});
    const auto expected = read_bytes(existing);
    assert(!write(existing.u8string()));
    assert(read_bytes(existing) == expected);
    assert(!std::filesystem::exists(missing));
    assert(!write(missing.u8string()));
    assert(!std::filesystem::exists(missing));
}

void test_rejected_writes_preserve_destination(const std::filesystem::path& directory) {
    for (const auto& invalid : {WavData{{0.5f}, 0}, WavData{{0.5f}, -16000}, WavData{{}, 16000}}) {
        check_rejected_write(directory, [&](const std::string& path) { return invalid.save(path); });
        check_rejected_write(directory, [&](const std::string& path) { return invalid.save(path.c_str()); });
        check_rejected_write(directory, [&](const std::string& path) {
            return invalid.save(std::filesystem::u8path(path));
        });
        check_rejected_write(directory, [&](const std::string& path) {
            return WavData::write_mono(path, invalid.samples, invalid.sample_rate);
        });
        check_rejected_write(directory, [&](const std::string& path) {
            return speech_core::write_wav_mono_pcm16(path, invalid.samples, invalid.sample_rate);
        });
        check_rejected_write(directory, [&](const std::string& path) {
            return speech_core::write_wav_mono_pcm16(
                path, invalid.samples.data(), invalid.samples.size(), invalid.sample_rate);
        });

        std::ostringstream stream;
        stream << "existing bytes";
        assert(!invalid.save(stream));
        assert(stream.str() == "existing bytes");
    }
    check_rejected_write(directory, [](const std::string& path) {
        return speech_core::write_wav_mono_pcm16(path, nullptr, 1, 16000);
    });
}

void test_stream_save_and_errors(const std::filesystem::path& path) {
    const WavData input{{-2, -0.5f, 0, 0.5f, 2}, 24000};
    wav_test::write_fixture(path, 1, 16, 1, 24000, {0x8001, 0xc001, 0, 0x3fff, 0x7fff});
    const auto expected = read_bytes(path);
    std::ostringstream encoded(std::ios::out | std::ios::binary);
    assert(input.save(encoded));
    assert(encoded.str() == expected);

    std::ostringstream raw(std::ios::out | std::ios::binary);
    assert(WavData::write_mono(raw, input.samples.data(), input.samples.size(), input.sample_rate));
    assert(raw.str() == expected);
    std::istringstream encoded_input(encoded.str(), std::ios::in | std::ios::binary);
    WavData decoded;
    assert(decoded.load_mono(encoded_input));
    assert(decoded.sample_rate == input.sample_rate);
    assert(decoded.samples.size() == input.samples.size());

    std::ostringstream failed_output;
    failed_output.setstate(std::ios::badbit);
    assert(!input.save(failed_output));
    assert(failed_output.str().empty());

    std::istringstream failed_input(expected);
    failed_input.setstate(std::ios::badbit);
    assert(!decoded.load_mono(failed_input));
    decoded = input;
    std::istringstream truncated(expected.substr(0, expected.size() - 2));
    assert(!decoded.load_mono(truncated));
}

}  // namespace

int main() {
    wav_test::TemporaryDirectory directory("speech_core_test_wav_io");
    const auto path = directory.path / "fixture.wav";
    test_formats_and_downmix(path);
    test_utf8_write_and_read(directory.path);
    test_invalid_input(path);
    test_path_overloads_and_legacy_api(directory.path);
    test_rejected_writes_preserve_destination(directory.path);
    test_stream_save_and_errors(path);
    std::puts("All WAV I/O tests passed.");
    return 0;
}
