#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace speech_core {

/// Minimal SentencePiece-compatible tokenizer used by the Pocket TTS ONNX
/// bundle. The export stores its vocabulary and unigram scores as JSON so the
/// Android runtime does not need to link libsentencepiece.
class PocketTtsTokenizer {
public:
    PocketTtsTokenizer(const std::string& vocab_json,
                       const std::string& token_scores_json);
    ~PocketTtsTokenizer();

    PocketTtsTokenizer(PocketTtsTokenizer&&) noexcept;
    PocketTtsTokenizer& operator=(PocketTtsTokenizer&&) noexcept;

    PocketTtsTokenizer(const PocketTtsTokenizer&) = delete;
    PocketTtsTokenizer& operator=(const PocketTtsTokenizer&) = delete;

    std::vector<std::int32_t> encode_ids(const std::string& text) const;
    std::vector<std::string> encode_tokens(const std::string& text) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace speech_core
