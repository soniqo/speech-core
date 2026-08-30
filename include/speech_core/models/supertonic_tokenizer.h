#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace speech_core {

/// One synthesis piece produced by `SupertonicTokenizer::fit_to_window()`.
struct SupertonicPiece {
    std::string text;                 ///< raw (not yet preprocessed) text of the piece
    bool        continuation = false; ///< the sentence carries on in the next piece: the front-end
                                      ///  terminates it with "," (continuation intonation), not "."
    bool        pause_before = false; ///< a sentence boundary separates this piece from the previous
                                      ///  one (host may insert its inter-chunk silence); false at a
                                      ///  forced intra-sentence split and for the first piece
};

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

    /// Construct from an in-memory `codepoint → id` table (the contents of unicode_indexer.json).
    explicit SupertonicTokenizer(std::vector<int32_t> indexer);

    /// Whether `lang` (ISO code, e.g. "de", "ko") is in Supertonic's AVAILABLE_LANGS.
    bool supports(const std::string& lang) const;

    /// Split free-form text into per-synthesis chunks. Mirrors `helper.py::_chunk_text`: sentences
    /// (terminal punctuation followed by whitespace) are packed greedily up to `max_codepoints`
    /// raw codepoints (0 ⇒ the model's text capacity). A sentence longer than that budget is
    /// **not** word-packed at the budget — that strands its last word or two in a tiny chunk —
    /// it is emitted whole so the caller can measure its real duration (`fit_to_window`) rather
    /// than guess from a character count. Every emitted chunk fits the model's text capacity
    /// after NFKD and the `<lang>` wrap (`wrapped_length() <= max_text_tokens()`); text that
    /// cannot is split into balanced pieces at the best sentence/clause/word boundary.
    /// Throws std::invalid_argument on an unsupported language.
    std::vector<std::string> chunk(const std::string& text, const std::string& lang,
                                   int max_codepoints = 0) const;

    /// Fit one chunk into a fixed-length latent window. `latent_frames(text, continuation)` runs
    /// the duration predictor on a candidate and returns its latent length in frames. While a
    /// candidate exceeds `max_frames` it is split into two balanced pieces at the best boundary
    /// (sentence > clause > word), each measured again. Pieces shorter than `min_codepoints`
    /// are never produced (every piece costs a full fixed-shape graph run, so a two-word fragment
    /// is not worth one); after `max_depth` splits, or when no split is possible, the overflowing
    /// candidate is returned as-is for the caller to truncate.
    static std::vector<SupertonicPiece> fit_to_window(
        const std::string& chunk,
        const std::function<int(const std::string& text, bool continuation)>& latent_frames,
        int max_frames, int min_codepoints = 12, int max_depth = 6);

    /// Result of tokenizing one chunk for the fixed-T graphs.
    struct Tokens {
        std::vector<int32_t> ids;   ///< length == text_t (zero-padded)
        std::vector<float>   mask;  ///< length == text_t (1.0 real, 0.0 pad), feeds text_mask[1,1,T]
    };

    /// Full front-end for one chunk: NFKD + cleanup + `<lang>` wrap + tokenize, then right-pad ids
    /// to `text_t` (with 0) and build the float mask. A chunk without terminal punctuation gets
    /// "." appended — or "," when `continuation` is set, so a fragment that continues in the next
    /// chunk is rendered with continuation rather than terminal intonation.
    /// Throws std::invalid_argument on bad language.
    Tokens process(const std::string& text, const std::string& lang, int text_t = 128,
                   bool continuation = false) const;

    /// Token count of the wrapped, preprocessed form of `text` (what `process()` emits before
    /// padding; it must not exceed `max_text_tokens()` or `process()` truncates).
    int wrapped_length(const std::string& text, const std::string& lang) const;

    /// Largest wrapped+tokenized length the front-end will emit before padding (== text_t).
    int max_text_tokens() const { return max_text_tokens_; }

private:
    std::string preprocess(const std::string& text, const std::string& lang,
                           bool continuation) const;  // NFKD + clean + wrap
    int32_t     lookup(uint32_t codepoint) const;
    void        emit_within_capacity(const std::string& text, const std::string& lang,
                                     std::vector<std::string>& out) const;

    std::vector<int32_t> indexer_;          // [65536]
    int                  max_text_tokens_ = 128;
};

}  // namespace speech_core
