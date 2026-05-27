#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace speech_core {

/// VoxCPM2 BPE tokenizer.
///
/// Loads `tokenizer.json` (HuggingFace `tokenizers` format) and reproduces the
/// reference encode / decode pipeline:
///
///   encode(text):
///     1. Normalize: prepend ▁, replace each ' ' with ▁ (SentencePiece-style)
///     2. BPE on Unicode codepoints with byte fallback (<0xXX> for codepoints
///        absent from the vocab)
///     3. Post-process: prepend BOS (<s>, id=1)
///
///   decode(ids):
///     1. Skip special tokens
///     2. Join token strings
///     3. Replace ▁ with space, collapse consecutive <0xXX> back into raw bytes
///     4. Strip the single leading space introduced by the normalizer
///
/// Hand-rolled instead of pulling tokenizers-cpp because the latter brings a
/// Rust toolchain dependency that breaks our Android cross-compile and Yocto
/// build paths. The tokenizer model is fixed (VoxCPM2 doesn't update it
/// across releases), so a targeted reimplementation has the same correctness
/// profile as the wrapper would, at zero extra build cost.
class VoxCPM2Tokenizer {
public:
    explicit VoxCPM2Tokenizer(const std::string& tokenizer_json_path);

    /// Encode UTF-8 text → token IDs, including the BOS prefix.
    std::vector<int> encode(const std::string& text) const;

    /// Decode token IDs → UTF-8 text. Special tokens are skipped.
    std::string decode(const std::vector<int>& ids) const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int unk_id() const { return unk_id_; }

    /// Look up the ID for a specific token (e.g. "<|audio_start|>"). Returns -1
    /// if the token isn't in the vocabulary.
    int token_id(const std::string& token) const;

    size_t vocab_size() const { return id_to_token_.size(); }

private:
    // vocab_["▁hello"] = 31986;  id_to_token_[31986] = "▁hello"
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string>             id_to_token_;

    // merge_rank_["▁ t"] = 4  → merge ("▁","t") at priority 4 (lower = first)
    std::unordered_map<std::string, int> merge_rank_;

    // Special tokens are skipped during decode and never participate in BPE.
    std::unordered_set<int> special_ids_;

    int  bos_id_         = 1;
    int  eos_id_         = 2;
    int  unk_id_         = 0;
    bool byte_fallback_  = false;

    // Cached <0x00>..<0xFF> → ID (only populated if byte_fallback_ is true).
    std::vector<int> byte_token_ids_;  // index by byte value, -1 if missing

    // BPE engine — operates on already-normalized text.
    std::vector<int> bpe(const std::string& word) const;
};

}  // namespace speech_core
