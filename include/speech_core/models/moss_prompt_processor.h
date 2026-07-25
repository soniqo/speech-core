#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace speech_core {

struct MossPreparedPrompt {
    std::vector<int64_t> input_ids;
    std::size_t audio_placeholder_count = 0;
    int64_t eos_token_id = 151645;
};

/// Published MOSS chat prompt and audio-placeholder expansion.
///
/// The text instruction and its token ids are pinned to the source revision
/// named by the portable model bundle. Keeping the already-tokenized fixed
/// prompt here avoids embedding a full BPE encoder in native inference; model
/// output still uses the bundle's vocabulary for decoding.
class MossPromptProcessor {
public:
    static constexpr int64_t kAudioTokenId = 151671;
    static constexpr int64_t kEosTokenId = 151645;
    static constexpr double kAudioTokensPerSecond = 12.5;
    static constexpr int kTimeMarkerEverySeconds = 5;

    static MossPreparedPrompt prepare(std::size_t audio_token_count);
    static std::vector<int64_t> make_audio_span(
        std::size_t audio_token_count);
};

/// Qwen byte-level output decoder for the MOSS bundle's `vocab.json`.
class MossTokenizerDecoder {
public:
    explicit MossTokenizerDecoder(const std::string& vocab_json_path);

    /// Decode generated ids exactly as the Qwen tokenizer's byte decoder.
    /// Added control ids are absent from vocab.json and are skipped.
    std::string decode(const std::vector<int64_t>& ids) const;

    std::size_t vocab_size() const { return tokens_by_id_.size(); }

private:
    std::vector<std::string> tokens_by_id_;
    int inverse_byte_map_[512] = {};
};

}  // namespace speech_core
