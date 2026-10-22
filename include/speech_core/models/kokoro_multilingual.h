#pragma once

#include <string>
#include <unordered_map>

namespace speech_core::multilingual {

/// Non-English phonemizers for Kokoro TTS.
///
/// Dictionary-first approach with rule-based G2P fallback:
/// 1. Split text into words (whitespace + punctuation boundaries)
/// 2. For each word: try dictionary lookup (lowercase), if found use it
/// 3. If not found: apply rule-based grapheme-to-phoneme conversion
/// 4. Pass punctuation tokens through
///
/// Languages:
/// - French, Spanish, Portuguese, Italian, Hindi — dictionary + rule-based
/// - German — number/abbreviation normalization + dictionary (dict_de.json)
///   + inflection/compound fallbacks + rule-based G2P in the espeak-ng
///   convention of the Kokoro German voices (see kokoro_german.h)
/// - Japanese — Katakana/Hiragana tables, kanji passthrough
/// - Chinese — Pinyin->IPA conversion (requires pre-segmented pinyin input)

// --- Dictionary-first phonemizers (preferred entry points) ---

std::string french_phonemize(const std::string& text,
    const std::unordered_map<std::string, std::string>& dict);

std::string spanish_phonemize(const std::string& text,
    const std::unordered_map<std::string, std::string>& dict);

std::string italian_phonemize(const std::string& text,
    const std::unordered_map<std::string, std::string>& dict);

std::string portuguese_phonemize(const std::string& text,
    const std::unordered_map<std::string, std::string>& dict);

std::string hindi_phonemize(const std::string& text,
    const std::unordered_map<std::string, std::string>& dict);

/// German: normalizes numbers, dates and abbreviations first, then resolves
/// each word through the dictionary, its inflected and compound forms, and
/// finally the rule-based G2P. Never applies the ʁ -> ɹ post-processing the
/// Romance languages use: the German voices were trained with espeak's r/ɾ.
std::string german_phonemize(const std::string& text,
    const std::unordered_map<std::string, std::string>& dict);

std::string japanese_phonemize(const std::string& text);

std::string chinese_phonemize(const std::string& text);

// --- Rule-based G2P fallback (used when word not in dictionary) ---

std::string french_g2p(const std::string& text);
std::string spanish_g2p(const std::string& text);
std::string portuguese_g2p(const std::string& text);
std::string italian_g2p(const std::string& text);
std::string japanese_g2p(const std::string& text);
std::string chinese_g2p(const std::string& text);
std::string hindi_g2p(const std::string& text);

}  // namespace speech_core::multilingual
