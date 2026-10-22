#pragma once

#include <string>
#include <unordered_map>

namespace speech_core::german {

/// German front-end for the Kokoro German voices (kikiri-tts fine-tunes).
///
/// The voices were trained on espeak-ng 1.52 German phonemes as emitted by
/// misaki's EspeakG2P (diphthongs folded to I/W, affricates to ʦ/ʧ/ʤ, the
/// short ü folded to y). This module reproduces that convention without
/// espeak:
///
///   1. normalize_text(): numbers, ordinals, dates, times, currency, units,
///      percentages and common abbreviations are spelled out.
///   2. phonemize(): dictionary lookup (dict_de.json), then an inflected
///      form of a dictionary word, then a compound of dictionary words, then
///      the rule-based g2p_word() for anything else.
///
/// All functions are pure and thread-safe.

using Dictionary = std::unordered_map<std::string, std::string>;

/// Spell out digits, dates, times, currency amounts, units and abbreviations.
std::string normalize_text(const std::string& text);

/// Rule-based grapheme-to-phoneme conversion of one word (no dictionary).
/// The word is lower-cased and accent-folded; the result carries primary (ˈ)
/// and secondary (ˌ) stress marks placed before the stressed vowel.
std::string g2p_word(const std::string& word);

/// Phonemes for one word: dictionary, inflected stem, compound, then rules.
/// [word] keeps its original case: a capitalised homograph such as "Weg"
/// selects the noun reading.
std::string phonemize_word(const std::string& word, const Dictionary& dict);

/// Phonemize already-normalized text. Words are separated by single spaces;
/// sentence punctuation (, . ! ? ; : - " ( )) passes through.
std::string phonemize(const std::string& text, const Dictionary& dict);

}  // namespace speech_core::german
