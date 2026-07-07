#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// Indic-Mio (Qwen3-family) byte-level BPE tokenizer.
///
/// Loads `tokenizer.json` (HuggingFace `tokenizers` format) and reproduces the
/// Qwen2/Qwen3 encode pipeline, which differs from every other tokenizer in
/// this tree (VoxCPM2/Supertonic are SentencePiece-style, Chatterbox is
/// Whitespace-BPE):
///
///   encode(text):
///     1. Pretokenize with the Qwen Split regex
///        (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
///        ' ?[^\s\p{L}\p{N}]+[\r\n]*' | \s*[\r\n]+ | \s+(?!\S) | \s+
///        implemented as a hand-rolled scanner over codepoints with generated
///        Unicode \p{L}/\p{N} range tables (std::regex has no \p classes).
///     2. ByteLevel: map each pretoken's UTF-8 bytes through the GPT-2
///        byte→unicode table (space→Ġ etc.).
///     3. BPE over the mapped symbols using model.merges ranks.
///
/// Correctness is pinned by golden fixtures generated from the reference HF
/// tokenizer over the product domain (Devanagari + emotion tags + English +
/// digits + the full chat prompt) — see test_indic_mio_tokenizer.cpp. The
/// upstream normalizer is NFC; this implementation does NOT re-normalize and
/// expects NFC input (standard for text typed in the Studio UI; documented
/// limitation for NFD sources).
///
/// encode() never emits added/special tokens — hosts insert chat-control ids
/// (<|im_start|> 151644, <|im_end|> 151645, <|endoftext|> 151643) by id,
/// which reproduces HF's added-token splitting for the fixed prompt template.
class IndicMioTokenizer {
public:
    explicit IndicMioTokenizer(const std::string& tokenizer_json_path);

    /// Encode plain UTF-8 text (no added-token handling) → token ids.
    std::vector<int> encode(const std::string& text) const;

    size_t vocab_size() const { return vocab_.size(); }

    /// Exposed for tests: pretoken byte ranges of `text` under the Qwen regex.
    std::vector<std::pair<size_t, size_t>> pretokenize(const std::string& text) const;

private:
    std::vector<int> bpe(const std::string& mapped) const;

    std::unordered_map<std::string, int> vocab_;       // byte-level token → id
    std::unordered_map<std::string, int> merge_rank_;  // "left\x1Fright" → rank
    std::string byte_enc_[256];                        // raw byte → mapped UTF-8
};

}  // namespace speech_core
