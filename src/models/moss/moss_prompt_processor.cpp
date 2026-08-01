#include "speech_core/models/moss_prompt_processor.h"

#include "speech_core/util/json.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace speech_core {
namespace {

constexpr std::array<int64_t, 15> kPromptPrefix = {
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645,
    198, 151644, 872, 198, 151669,
};

constexpr std::array<int64_t, 71> kPromptSuffix = {
    151670, 198, 14880, 44063, 111268, 46670, 61443, 17714, 108704,
    3837, 73157, 104383, 58362, 23031, 71618, 26606, 20450, 111420,
    33108, 104283, 17340, 72640, 9909, 58, 50, 15, 16, 60, 5373, 58,
    50, 15, 17, 60, 5373, 58, 50, 15, 18, 60, 1940, 7552, 111749,
    3837, 110644, 17714, 110019, 105761, 43815, 90395, 18493, 37474,
    100072, 111066, 80565, 20450, 111420, 3837, 23031, 104542, 117932,
    75882, 37474, 105761, 101121, 1773, 151645, 198, 151644, 77091, 198,
};

std::string trim_ascii(std::string value) {
    auto not_space = [](unsigned char character) {
        return !std::isspace(character);
    };
    const auto first = std::find_if(value.begin(), value.end(), not_space);
    const auto last = std::find_if(
        value.rbegin(), value.rend(), not_space).base();
    if (first >= last) return {};
    return std::string(first, last);
}

void append_utf8_codepoint(
    const std::string& value, std::size_t& offset, uint32_t& codepoint) {
    const unsigned char first =
        static_cast<unsigned char>(value[offset++]);
    if ((first & 0x80u) == 0) {
        codepoint = first;
        return;
    }
    int continuation_count = 0;
    if ((first & 0xe0u) == 0xc0u) {
        codepoint = first & 0x1fu;
        continuation_count = 1;
    } else if ((first & 0xf0u) == 0xe0u) {
        codepoint = first & 0x0fu;
        continuation_count = 2;
    } else if ((first & 0xf8u) == 0xf0u) {
        codepoint = first & 0x07u;
        continuation_count = 3;
    } else {
        codepoint = first;
        return;
    }
    if (offset + static_cast<std::size_t>(continuation_count)
            > value.size()) {
        codepoint = first;
        return;
    }
    for (int index = 0; index < continuation_count; ++index) {
        const unsigned char next =
            static_cast<unsigned char>(value[offset++]);
        if ((next & 0xc0u) != 0x80u) {
            codepoint = first;
            return;
        }
        codepoint = (codepoint << 6u) | (next & 0x3fu);
    }
}

}  // namespace

std::vector<int64_t> MossPromptProcessor::make_audio_span(
    std::size_t audio_token_count) {
    if (audio_token_count == 0) return {};

    const int tokens_per_marker = static_cast<int>(
        kAudioTokensPerSecond
        * static_cast<double>(kTimeMarkerEverySeconds));
    const double duration =
        static_cast<double>(audio_token_count) / kAudioTokensPerSecond;

    std::vector<int64_t> output;
    output.reserve(audio_token_count + 16);
    std::size_t consumed = 0;
    for (int seconds = kTimeMarkerEverySeconds;
         seconds <= static_cast<int>(duration);
         seconds += kTimeMarkerEverySeconds) {
        const std::size_t position = static_cast<std::size_t>(
            (seconds / kTimeMarkerEverySeconds) * tokens_per_marker);
        if (position > consumed) {
            output.insert(
                output.end(), position - consumed, kAudioTokenId);
            consumed = position;
        }
        for (const char digit : std::to_string(seconds)) {
            output.push_back(15 + static_cast<int64_t>(digit - '0'));
        }
    }
    if (audio_token_count > consumed) {
        output.insert(
            output.end(), audio_token_count - consumed, kAudioTokenId);
    }
    return output;
}

MossPreparedPrompt MossPromptProcessor::prepare(
    std::size_t audio_token_count) {
    if (audio_token_count == 0) {
        throw std::invalid_argument(
            "MOSS prompt requires at least one audio token");
    }
    const std::vector<int64_t> audio_span =
        make_audio_span(audio_token_count);

    MossPreparedPrompt result;
    result.eos_token_id = kEosTokenId;
    result.audio_placeholder_count = static_cast<std::size_t>(
        std::count(audio_span.begin(), audio_span.end(), kAudioTokenId));
    result.input_ids.reserve(
        kPromptPrefix.size() + audio_span.size() + kPromptSuffix.size());
    result.input_ids.insert(
        result.input_ids.end(), kPromptPrefix.begin(), kPromptPrefix.end());
    result.input_ids.insert(
        result.input_ids.end(), audio_span.begin(), audio_span.end());
    result.input_ids.insert(
        result.input_ids.end(), kPromptSuffix.begin(), kPromptSuffix.end());
    return result;
}

MossTokenizerDecoder::MossTokenizerDecoder(
    const std::string& vocab_json_path) {
    const std::string text = json::read_file(vocab_json_path);
    if (text.empty()) {
        throw std::runtime_error(
            "Unable to read MOSS vocabulary: " + vocab_json_path);
    }
    const auto vocabulary = json::parse_vocab_index(text);
    if (vocabulary.empty()) {
        throw std::runtime_error(
            "MOSS vocabulary contains no usable tokens: "
            + vocab_json_path);
    }

    int maximum_id = -1;
    for (const auto& entry : vocabulary) {
        maximum_id = std::max(maximum_id, entry.second);
    }
    tokens_by_id_.resize(static_cast<std::size_t>(maximum_id + 1));
    for (const auto& entry : vocabulary) {
        if (entry.second >= 0) {
            tokens_by_id_[static_cast<std::size_t>(entry.second)] =
                entry.first;
        }
    }

    std::fill(
        std::begin(inverse_byte_map_), std::end(inverse_byte_map_), -1);
    bool direct[256] = {};
    for (int value = static_cast<int>('!');
         value <= static_cast<int>('~'); ++value) {
        direct[value] = true;
    }
    for (int value = 0xa1; value <= 0xac; ++value) direct[value] = true;
    for (int value = 0xae; value <= 0xff; ++value) direct[value] = true;
    int replacement = 0;
    for (int byte = 0; byte < 256; ++byte) {
        const int codepoint = direct[byte] ? byte : 256 + replacement++;
        inverse_byte_map_[codepoint] = byte;
    }
}

std::string MossTokenizerDecoder::decode(
    const std::vector<int64_t>& ids) const {
    std::string output;
    for (const int64_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= tokens_by_id_.size()) {
            continue;
        }
        const std::string& token =
            tokens_by_id_[static_cast<std::size_t>(id)];
        std::size_t offset = 0;
        while (offset < token.size()) {
            uint32_t codepoint = 0;
            const std::size_t previous = offset;
            append_utf8_codepoint(token, offset, codepoint);
            if (codepoint < 512 && inverse_byte_map_[codepoint] >= 0) {
                output.push_back(static_cast<char>(
                    inverse_byte_map_[codepoint]));
            } else {
                output.append(token, previous, offset - previous);
            }
        }
    }
    return trim_ascii(std::move(output));
}

}  // namespace speech_core
