#include "speech_core/util/text_chunker.h"

namespace speech_core {

namespace {

bool is_sentence_end(char c) { return c == '.' || c == '!' || c == '?'; }
bool is_clause_end(char c) { return c == ',' || c == ';' || c == ':'; }

std::string trimmed(const std::string& s) {
    const char* ws = " \t\r\n";
    size_t begin = s.find_first_not_of(ws);
    if (begin == std::string::npos) return "";
    size_t end = s.find_last_not_of(ws);
    return s.substr(begin, end - begin + 1);
}

// Split into units, keeping end-punctuation attached to the preceding unit.
// Newlines are treated as boundaries too, so list-style text splits cleanly.
std::vector<std::string> split_units(const std::string& text,
                                     bool (*is_end)(char)) {
    std::vector<std::string> units;
    std::string current;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if (c == '\n') {
            auto t = trimmed(current);
            if (!t.empty()) units.push_back(t);
            current.clear();
            continue;
        }
        current.push_back(c);
        if (is_end(c)) {
            // Absorb runs of enders and a trailing closing quote so "..." or
            // '?!"' stays with its sentence.
            while (i + 1 < text.size() &&
                   (is_end(text[i + 1]) || text[i + 1] == '"' ||
                    text[i + 1] == '\'')) {
                current.push_back(text[++i]);
            }
            auto t = trimmed(current);
            if (!t.empty()) units.push_back(t);
            current.clear();
        }
    }
    auto t = trimmed(current);
    if (!t.empty()) units.push_back(t);
    return units;
}

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::string current;
    for (char c : text) {
        if (c == ' ' || c == '\t') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) words.push_back(current);
    return words;
}

size_t utf8_char_len(unsigned char byte) {
    if ((byte & 0x80) == 0x00) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 1;  // invalid lead byte — advance one byte, never split worse
}

// Last resort for a single word the budget cannot hold: cut on UTF-8
// character boundaries.
std::vector<std::string> split_chars(const std::string& word,
                                     const TokenCounter& count,
                                     size_t max_tokens) {
    std::vector<std::string> parts;
    std::string current;
    size_t i = 0;
    while (i < word.size()) {
        size_t len = utf8_char_len(static_cast<unsigned char>(word[i]));
        if (i + len > word.size()) len = word.size() - i;
        std::string candidate = current + word.substr(i, len);
        if (!current.empty() && count(candidate) > max_tokens) {
            parts.push_back(current);
            current = word.substr(i, len);
        } else {
            current = std::move(candidate);
        }
        i += len;
    }
    if (!current.empty()) parts.push_back(current);
    return parts;
}

// Greedily pack `units` into chunks of at most `max_tokens`; a unit that is
// alone too large is refined at the next split level.
void pack_units(const std::vector<std::string>& units,
                const TokenCounter& count, size_t max_tokens, int level,
                std::vector<std::string>& out) {
    std::string acc;
    auto flush = [&]() {
        if (!acc.empty()) {
            out.push_back(acc);
            acc.clear();
        }
    };
    for (const auto& unit : units) {
        std::string candidate = acc.empty() ? unit : acc + " " + unit;
        if (count(candidate) <= max_tokens) {
            acc = std::move(candidate);
            continue;
        }
        flush();
        if (count(unit) <= max_tokens) {
            acc = unit;
            continue;
        }
        // The unit alone exceeds the budget — refine it.
        std::vector<std::string> finer;
        if (level == 0) {
            finer = split_units(unit, is_clause_end);
        } else if (level == 1) {
            finer = split_words(unit);
        }
        if (level >= 2 || finer.size() <= 1) {
            for (auto& part : split_chars(unit, count, max_tokens)) {
                out.push_back(std::move(part));
            }
        } else {
            pack_units(finer, count, max_tokens, level + 1, out);
        }
    }
    flush();
}

}  // namespace

std::vector<std::string> chunk_text_for_synthesis(
    const std::string& text, const TokenCounter& count_tokens,
    size_t max_tokens, size_t hard_cap_tokens, size_t min_tail_tokens) {
    std::vector<std::string> chunks;
    if (max_tokens == 0) return chunks;

    pack_units(split_units(text, is_sentence_end), count_tokens, max_tokens,
               0, chunks);

    // Merge an unreliably-small tail into its predecessor when the model's
    // hard input capacity allows it.
    if (chunks.size() >= 2 &&
        count_tokens(chunks.back()) < min_tail_tokens) {
        std::string merged = chunks[chunks.size() - 2] + " " + chunks.back();
        if (count_tokens(merged) <= hard_cap_tokens) {
            chunks.pop_back();
            chunks.back() = std::move(merged);
        }
    }
    return chunks;
}

}  // namespace speech_core
