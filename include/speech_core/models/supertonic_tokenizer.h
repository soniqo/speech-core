#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace speech_core {

/// SupertonicTTS G2P-free text front-end — the part that, for Kokoro/espeak-class TTS, is the
/// hardest and most GPL-entangled piece, and which Supertonic collapses to:
/// **NFKD + regex cleanup + `<lang>…</lang>` wrap + codepoint→token-id table lookup.**
/// No phonemizer, no IPA, no lexicon, no espeak.
///
/// Faithful C++ port of `Supertone/supertonic` `py/helper.py::UnicodeProcessor` (MIT); validated
/// against it in the Runner repo's `speech-models/stmodels/infer.py`. Two traps it handles:
///   - **codepoint (not byte) work** — everything operates on decoded UTF-32, never `std::string`
///     byte length.
///   - **unknown token = -1** — out-of-range / unmapped codepoints resolve to -1 (never an OOB
///     table index). The model masks padded positions; -1 ids only appear at real positions for
///     genuinely out-of-vocab characters.
///
/// NFKD is the keystone: it decomposes precomposed Latin accents (ä → a +◌̈) and Hangul syllables
/// (한 → ㅎ+ㅏ+ㄴ) into in-vocab components — which is exactly why German and Korean need no
/// special-casing. NFKD here is provided by utf8proc (`UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT`).
class SupertonicTokenizer {
public:
    /// Token id for codepoints absent from the indexer table.
    static constexpr int32_t kUnknownId = -1;

    /// @param unicode_indexer_path flat JSON array of 65536 ints (`codepoint → id`, -1 if unsupported)
    /// @param tts_json_path        `tts.json` (read for forward-compat; AVAILABLE_LANGS is baked in)
    explicit SupertonicTokenizer(const std::string& unicode_indexer_path,
                                 const std::string& tts_json_path = {});

    /// Whether `lang` (ISO code, e.g. "de", "ko") is in Supertonic's AVAILABLE_LANGS.
    bool supports(const std::string& lang) const;

    /// Split free-form text into per-synthesis chunks. Caps each chunk so the wrapped, tokenized
    /// form fits the exported fixed text length (`max_text_tokens`, default 128). Sentence-aware,
    /// codepoint-counted; CJK uses a tighter cap. Mirrors `helper.py::_chunk_text`.
    std::vector<std::string> chunk(const std::string& text, const std::string& lang) const;

    /// Result of tokenizing one chunk for the fixed-T graphs.
    struct Tokens {
        std::vector<int32_t> ids;   ///< length == text_t (zero-padded)
        std::vector<float>   mask;  ///< length == text_t (1.0 real, 0.0 pad), feeds text_mask[1,1,T]
    };

    /// Full front-end for one chunk: NFKD + cleanup + `<lang>` wrap + tokenize, then right-pad ids
    /// to `text_t` (with 0) and build the float mask. Throws std::invalid_argument on bad language.
    Tokens process(const std::string& text, const std::string& lang, int text_t = 128) const;

    /// Largest wrapped+tokenized length the front-end will emit before padding (== text_t).
    int max_text_tokens() const { return max_text_tokens_; }

private:
    std::string preprocess(const std::string& text, const std::string& lang) const;  // NFKD + clean + wrap
    int32_t     lookup(uint32_t codepoint) const;

    std::vector<int32_t> indexer_;          // [65536]
    int                  max_text_tokens_ = 128;
};

}  // namespace speech_core
