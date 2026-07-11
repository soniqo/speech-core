#include "speech_core/util/text_chunker.h"

#include <algorithm>
#include <limits>
#include <tuple>

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
            // A long sentence with no comma/semicolon must still fall back
            // to whole words. Previously this skipped directly to UTF-8
            // character cuts, which could emit fragments such as
            // "voice-age" / "nt replies" for a normal hyphenated word.
            if (finer.size() <= 1) finer = split_words(unit);
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

std::vector<std::string> split_text_for_synthesis_retry(
    const std::string& text, const TokenCounter& count_tokens,
    size_t min_tokens, size_t max_tokens) {
    if (text.empty() || min_tokens == 0 || min_tokens > max_tokens) return {};

    std::vector<std::string> best;
    auto best_score = std::make_tuple(
        std::numeric_limits<int>::max(),
        std::numeric_limits<size_t>::max(),
        std::numeric_limits<size_t>::max());

    size_t split = 0;
    while (split < text.size()) {
        size_t len = utf8_char_len(static_cast<unsigned char>(text[split]));
        if (split + len > text.size()) len = text.size() - split;
        split += len;
        if (split >= text.size()) break;

        std::string left = trimmed(text.substr(0, split));
        std::string right = trimmed(text.substr(split));
        if (left.empty() || right.empty()) continue;

        const size_t left_tokens = count_tokens(left);
        const size_t right_tokens = count_tokens(right);
        if (left_tokens < min_tokens || right_tokens < min_tokens ||
            left_tokens > max_tokens || right_tokens > max_tokens) {
            continue;
        }

        int boundary_rank = 3;  // UTF-8 character boundary
        const char before = text[split - 1];
        const char after = text[split];

        // Treat a run such as `...`, `?!`, or `?!"` as one boundary. Never
        // leave part of that run at the beginning of the retry's right half.
        bool saw_sentence_end = false;
        bool saw_clause_end = false;
        size_t run_begin = split;
        while (run_begin > 0) {
            const char c = text[run_begin - 1];
            if (is_sentence_end(c)) {
                saw_sentence_end = true;
            } else if (is_clause_end(c)) {
                saw_clause_end = true;
            } else if (c != '\'' && c != '"') {
                break;
            }
            --run_begin;
        }
        if (saw_sentence_end || saw_clause_end) {
            if (is_sentence_end(after) || is_clause_end(after) ||
                after == '\'' || after == '"') {
                continue;
            }
            boundary_rank = saw_sentence_end ? 0 : 1;
        } else if (before == ' ' || before == '\t' ||
                   after == ' ' || after == '\t') {
            boundary_rank = 2;
        }

        const size_t largest = std::max(left_tokens, right_tokens);
        const size_t imbalance = left_tokens > right_tokens
            ? left_tokens - right_tokens
            : right_tokens - left_tokens;
        const auto score = std::make_tuple(boundary_rank, largest, imbalance);
        if (score < best_score) {
            best_score = score;
            best = {std::move(left), std::move(right)};
        }
    }
    return best;
}

}  // namespace speech_core
