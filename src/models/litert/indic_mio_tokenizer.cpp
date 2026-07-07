#include "speech_core/models/indic_mio_tokenizer.h"

#include "speech_core/util/json.h"

#include <utf8proc.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace speech_core {

namespace {

// --- JSON navigation over the flat parser (same helpers as the sibling
// tokenizers; util/json.h exposes skip_ws/parse_string/skip_value).
size_t find_member(const std::string& s, size_t i, const std::string& key) {
    // `i` must point just past the '{' of the object to search.
    while (i < s.size()) {
        ::json::skip_ws(s, i);
        if (i >= s.size() || s[i] == '}') return std::string::npos;
        if (s[i] == ',') { ++i; continue; }
        std::string k = ::json::parse_string(s, i);
        ::json::skip_ws(s, i);
        if (i < s.size() && s[i] == ':') ++i;
        ::json::skip_ws(s, i);
        if (k == key) return i;
        ::json::skip_value(s, i);
    }
    return std::string::npos;
}

int parse_int(const std::string& s, size_t& i) {
    bool neg = false;
    if (i < s.size() && s[i] == '-') { neg = true; ++i; }
    long long v = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') { v = v * 10 + (s[i] - '0'); ++i; }
    return static_cast<int>(neg ? -v : v);
}

// --- Unicode helpers (utf8proc — already a dependency of this target for the
// Supertonic tokenizer's NFKD; gives exact \p{L}/\p{N} category semantics).

bool is_letter(uint32_t cp) {
    switch (utf8proc_category(static_cast<utf8proc_int32_t>(cp))) {
        case UTF8PROC_CATEGORY_LU: case UTF8PROC_CATEGORY_LL:
        case UTF8PROC_CATEGORY_LT: case UTF8PROC_CATEGORY_LM:
        case UTF8PROC_CATEGORY_LO:
            return true;
        default:
            return false;
    }
}
bool is_number(uint32_t cp) {
    switch (utf8proc_category(static_cast<utf8proc_int32_t>(cp))) {
        case UTF8PROC_CATEGORY_ND: case UTF8PROC_CATEGORY_NL:
        case UTF8PROC_CATEGORY_NO:
            return true;
        default:
            return false;
    }
}

// NFC to match the tokenizer.json normalizer.
std::string nfc(const std::string& s) {
    utf8proc_uint8_t* out = nullptr;
    const utf8proc_ssize_t n = utf8proc_map(
        reinterpret_cast<const utf8proc_uint8_t*>(s.data()),
        static_cast<utf8proc_ssize_t>(s.size()), &out,
        static_cast<utf8proc_option_t>(UTF8PROC_COMPOSE | UTF8PROC_STABLE));
    if (n < 0 || !out) return s;  // invalid UTF-8 — tokenize as-is
    std::string result(reinterpret_cast<char*>(out), static_cast<size_t>(n));
    free(out);
    return result;
}

// Unicode White_Space (the \s class of the HF `tokenizers` regex engine).
bool is_space(uint32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
        case 0x85: case 0xA0: case 0x1680:
        case 0x2000: case 0x2001: case 0x2002: case 0x2003: case 0x2004:
        case 0x2005: case 0x2006: case 0x2007: case 0x2008: case 0x2009:
        case 0x200A: case 0x2028: case 0x2029: case 0x202F: case 0x205F:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

bool is_crlf(uint32_t cp) { return cp == 0x0A || cp == 0x0D; }

// Decode UTF-8 into codepoints, remembering each codepoint's byte offset.
void decode_utf8(const std::string& s, std::vector<uint32_t>& cps,
                 std::vector<size_t>& offsets) {
    size_t i = 0;
    while (i < s.size()) {
        offsets.push_back(i);
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = c;
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        if (i + len > s.size()) len = 1;
        for (size_t k = 1; k < len; ++k) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        }
        cps.push_back(cp);
        i += len;
    }
    offsets.push_back(s.size());
}

std::string encode_cp_utf8(uint32_t cp) {
    std::string out;
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

constexpr char kMergeSep = '\x1F';

}  // namespace

// ---------------------------------------------------------------------------
// Construction: parse model.vocab + model.merges, build the byte-level table.
// ---------------------------------------------------------------------------

IndicMioTokenizer::IndicMioTokenizer(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("IndicMioTokenizer: cannot open " + tokenizer_json_path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();

    size_t i = 0;
    ::json::skip_ws(s, i);
    if (i >= s.size() || s[i] != '{') {
        throw std::runtime_error("IndicMioTokenizer: tokenizer.json is not a JSON object");
    }
    ++i;
    const size_t model_pos = find_member(s, i, "model");
    if (model_pos == std::string::npos || s[model_pos] != '{') {
        throw std::runtime_error("IndicMioTokenizer: missing model object");
    }

    // model.vocab: { "token": id, ... }
    size_t vpos = find_member(s, model_pos + 1, "vocab");
    if (vpos == std::string::npos || s[vpos] != '{') {
        throw std::runtime_error("IndicMioTokenizer: missing model.vocab");
    }
    {
        size_t p = vpos + 1;
        vocab_.reserve(160000);
        while (p < s.size()) {
            ::json::skip_ws(s, p);
            if (p >= s.size() || s[p] == '}') break;
            if (s[p] == ',') { ++p; continue; }
            std::string tok = ::json::parse_string(s, p);
            ::json::skip_ws(s, p);
            if (p < s.size() && s[p] == ':') ++p;
            ::json::skip_ws(s, p);
            const int id = parse_int(s, p);
            vocab_.emplace(std::move(tok), id);
        }
    }

    // model.merges: [ ["a","b"], ... ]  (rank = array index)
    size_t mpos = find_member(s, model_pos + 1, "merges");
    if (mpos == std::string::npos || s[mpos] != '[') {
        throw std::runtime_error("IndicMioTokenizer: missing model.merges");
    }
    {
        size_t p = mpos + 1;
        int rank = 0;
        merge_rank_.reserve(160000);
        while (p < s.size()) {
            ::json::skip_ws(s, p);
            if (p >= s.size() || s[p] == ']') break;
            if (s[p] == ',') { ++p; continue; }
            if (s[p] != '[') {
                throw std::runtime_error("IndicMioTokenizer: merges entries must be arrays");
            }
            ++p;
            ::json::skip_ws(s, p);
            std::string a = ::json::parse_string(s, p);
            ::json::skip_ws(s, p);
            if (p < s.size() && s[p] == ',') ++p;
            ::json::skip_ws(s, p);
            std::string b = ::json::parse_string(s, p);
            ::json::skip_ws(s, p);
            if (p < s.size() && s[p] == ']') ++p;
            merge_rank_.emplace(a + kMergeSep + b, rank++);
        }
    }
    if (vocab_.empty() || merge_rank_.empty()) {
        throw std::runtime_error("IndicMioTokenizer: empty vocab or merges");
    }

    // GPT-2 byte→unicode table: printable ranges map to themselves, the rest
    // to U+0100.. in order.
    bool direct[256] = {false};
    for (int b = int('!'); b <= int('~'); ++b) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const uint32_t cp = direct[b] ? static_cast<uint32_t>(b)
                                      : static_cast<uint32_t>(256 + n++);
        byte_enc_[b] = encode_cp_utf8(cp);
    }
}

// ---------------------------------------------------------------------------
// Pretokenizer: the Qwen Split regex as a scanner over codepoints. Branches
// are tried in the regex's alternation order at each position.
// ---------------------------------------------------------------------------

std::vector<std::pair<size_t, size_t>> IndicMioTokenizer::pretokenize(
    const std::string& text) const {
    std::vector<uint32_t> cps;
    std::vector<size_t> off;
    decode_utf8(text, cps, off);
    const size_t n = cps.size();

    std::vector<std::pair<size_t, size_t>> out;
    size_t i = 0;
    auto lower = [](uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };

    while (i < n) {
        size_t j = i;

        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cps[i] == '\'' && i + 1 < n) {
            const uint32_t c1 = lower(cps[i + 1]);
            const uint32_t c2 = i + 2 < n ? lower(cps[i + 2]) : 0;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                j = i + 2;
            } else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                       (c1 == 'l' && c2 == 'l')) {
                j = i + 3;
            }
            if (j != i) {
                out.emplace_back(off[i], off[j]);
                i = j;
                continue;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+  (prefix branch tried first — greedy '?')
        {
            size_t k = i;
            if (!is_crlf(cps[k]) && !is_letter(cps[k]) && !is_number(cps[k]) &&
                k + 1 < n && is_letter(cps[k + 1])) {
                ++k;  // optional single non-letter prefix (space, mark, punct)
            }
            if (k < n && is_letter(cps[k])) {
                size_t e = k;
                while (e < n && is_letter(cps[e])) ++e;
                out.emplace_back(off[i], off[e]);
                i = e;
                continue;
            }
        }

        // 3. \p{N}  (single number codepoint on this checkpoint's regex)
        if (is_number(cps[i])) {
            out.emplace_back(off[i], off[i + 1]);
            ++i;
            continue;
        }

        // 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*'
        {
            size_t k = i;
            if (cps[k] == ' ' && k + 1 < n && !is_space(cps[k + 1]) &&
                !is_letter(cps[k + 1]) && !is_number(cps[k + 1])) {
                ++k;
            }
            if (k < n && !is_space(cps[k]) && !is_letter(cps[k]) && !is_number(cps[k])) {
                size_t e = k;
                while (e < n && !is_space(cps[e]) && !is_letter(cps[e]) &&
                       !is_number(cps[e])) ++e;
                while (e < n && is_crlf(cps[e])) ++e;
                out.emplace_back(off[i], off[e]);
                i = e;
                continue;
            }
        }

        // Whitespace branches share the maximal \s run from i.
        if (is_space(cps[i])) {
            size_t run_end = i;
            while (run_end < n && is_space(cps[run_end])) ++run_end;

            // 5. \s*[\r\n]+ — longest prefix of the run ending in CR/LF.
            size_t last_nl = std::string::npos;
            for (size_t k = i; k < run_end; ++k) {
                if (is_crlf(cps[k])) last_nl = k;
            }
            if (last_nl != std::string::npos) {
                out.emplace_back(off[i], off[last_nl + 1]);
                i = last_nl + 1;
                continue;
            }

            // 6. \s+(?!\S) — the whole run when at end-of-text, else run-1
            //    (leaving one space to prefix the next pretoken).
            if (run_end == n) {
                out.emplace_back(off[i], off[run_end]);
                i = run_end;
                continue;
            }
            if (run_end - i >= 2) {
                out.emplace_back(off[i], off[run_end - 1]);
                i = run_end - 1;
                continue;
            }

            // 7. \s+ — single space followed by non-space that no earlier
            //    branch claimed (e.g. before a digit).
            out.emplace_back(off[i], off[run_end]);
            i = run_end;
            continue;
        }

        // Unreachable for well-formed input: branch 4 accepts any non-space,
        // non-letter, non-number codepoint. Emit a single-cp token to be safe.
        out.emplace_back(off[i], off[i + 1]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// BPE over byte-level symbols.
// ---------------------------------------------------------------------------

std::vector<int> IndicMioTokenizer::bpe(const std::string& mapped) const {
    // Split the mapped string into single byte-level characters (each is one
    // UTF-8-encoded codepoint from the byte table).
    std::vector<std::string> word;
    {
        size_t i = 0;
        while (i < mapped.size()) {
            const unsigned char c = static_cast<unsigned char>(mapped[i]);
            size_t len = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : 3;
            if (i + len > mapped.size()) len = 1;
            word.emplace_back(mapped.substr(i, len));
            i += len;
        }
    }

    while (word.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_i = 0;
        for (size_t k = 0; k + 1 < word.size(); ++k) {
            const auto it = merge_rank_.find(word[k] + kMergeSep + word[k + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = k;
            }
        }
        if (best_rank == std::numeric_limits<int>::max()) break;
        word[best_i] += word[best_i + 1];
        word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
    }

    std::vector<int> ids;
    ids.reserve(word.size());
    for (const auto& tok : word) {
        const auto it = vocab_.find(tok);
        if (it == vocab_.end()) {
            throw std::runtime_error("IndicMioTokenizer: symbol not in vocab: " + tok);
        }
        ids.push_back(it->second);
    }
    return ids;
}

std::vector<int> IndicMioTokenizer::encode(const std::string& text) const {
    const std::string norm = nfc(text);
    std::vector<int> ids;
    for (const auto& [b, e] : pretokenize(norm)) {
        std::string mapped;
        mapped.reserve((e - b) * 2);
        for (size_t k = b; k < e; ++k) {
            mapped += byte_enc_[static_cast<unsigned char>(norm[k])];
        }
        const auto piece = bpe(mapped);
        ids.insert(ids.end(), piece.begin(), piece.end());
    }
    return ids;
}

}  // namespace speech_core
