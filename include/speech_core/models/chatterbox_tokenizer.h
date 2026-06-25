#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace speech_core {

/// Chatterbox multilingual text tokenizer.
///
/// Loads `grapheme_mtl_merged_expanded_v1.json` (HuggingFace `tokenizers` BPE,
/// vocab 2454 / 265 merges) + `Cangjie5_TC.json` and reproduces the reference
/// `MTLTokenizer.encode` pipeline:
///
///   1. punc_norm (capitalise, collapse spaces, replace LLM punctuation, ensure
///      ending punctuation)
///   2. lowercase + NFKD normalisation
///   3. per-language frontend:
///        ko -> Hangul -> Jamo (algorithmic)
///        zh -> Cangjie codes ([cj_*]) via Cangjie5_TC.json
///        ja/he/ru -> pass-through (matches the reference env where the optional
///        kakasi/dicta/russian-stress deps degrade to raw text)
///   4. prepend the `[lang]` token, replace ' ' with the `[SPACE]` token
///   5. extract added/special tokens greedily, Whitespace-pretokenize the gaps
///      (\w+ | [^\w\s]+), BPE each subword (byte-level char fallback -> [UNK])
///
/// The T3 BOT/EOT wrap (start_text_token 255, stop_text_token 0) is added by the
/// caller (the wrapper), not here — matching ChatterboxMultilingualTTS.generate.
///
/// Hand-rolled (no HuggingFace `tokenizers` Rust dep) to keep the Android/Yocto
/// cross-compile clean, same rationale as VoxCPM2Tokenizer.
class ChatterboxTokenizer {
public:
    /// `tokenizer_json` = grapheme_mtl_merged_expanded_v1.json. `cangjie_json`
    /// may be empty (Chinese then falls back to raw chars, as upstream does
    /// without the map).
    ChatterboxTokenizer(const std::string& tokenizer_json,
                        const std::string& cangjie_json = "");

    /// Encode UTF-8 text -> BPE token IDs for the given language (ISO code, e.g.
    /// "en", "ar", "zh"). Does NOT add BOT/EOT.
    std::vector<int> encode(const std::string& text, const std::string& lang) const;

    int token_id(const std::string& token) const;
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string>             id_to_token_;
    std::unordered_map<std::string, int> merge_rank_;   // "a b" -> rank
    // added/special token strings, longest-first for greedy matching
    std::vector<std::string>             added_;
    std::unordered_map<std::string, int> added_id_;
    std::unordered_map<std::string, std::string> cangjie_;  // glyph -> code
    int unk_id_ = 1;

    std::vector<int> bpe_word(const std::string& word) const;
    std::vector<int> tokenize_segment(const std::string& seg) const;
};

}  // namespace speech_core
