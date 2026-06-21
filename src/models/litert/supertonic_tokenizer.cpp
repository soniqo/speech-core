#include "speech_core/models/supertonic_tokenizer.h"

#include "speech_core/util/json.h"

#include <utf8proc.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace speech_core {
namespace {

// Authoritative AVAILABLE_LANGS from `supertonic/py/helper.py` / `stmodels/infer.py` — 32 entries.
// Note: includes "na", **excludes "zh"** (Chinese rides "ja"/codepoints). The <lang> tag is fed
// through the same codepoint tokenizer; there is no separate language tensor (tts.json: n_langs = 0).
const std::unordered_set<std::string>& available_langs() {
    static const std::unordered_set<std::string> kLangs = {
        "en","ko","ja","ar","bg","cs","da","de","el","es",
        "et","fi","fr","hi","hr","hu","id","it","lt","lv",
        "nl","pl","pt","ro","ru","sk","sl","sv","tr","uk",
        "vi","na",
    };
    return kLangs;
}

// ---- UTF-8 <-> UTF-32 (codepoint-level work; never byte length) ----
std::vector<char32_t> utf8_to_u32(const std::string& s) {
    std::vector<char32_t> out;
    out.reserve(s.size());
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        char32_t cp; int len;
        if      ((c & 0x80) == 0x00) { cp = c;        len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { out.push_back(0xFFFD); ++i; continue; }
        if (i + static_cast<size_t>(len) > n) { out.push_back(0xFFFD); break; }
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { out.push_back(0xFFFD); ++i; continue; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

void append_u32(std::string& s, char32_t cp) {
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string u32_to_utf8(const std::vector<char32_t>& v) {
    std::string s;
    s.reserve(v.size() * 2);
    for (char32_t cp : v) append_u32(s, cp);
    return s;
}

// NFKD via utf8proc (decompose + compatibility). The keystone of the G2P-free front-end.
std::string nfkd(const std::string& s) {
    utf8proc_uint8_t* out = nullptr;
    utf8proc_ssize_t n = utf8proc_map(
        reinterpret_cast<const utf8proc_uint8_t*>(s.data()),
        static_cast<utf8proc_ssize_t>(s.size()), &out,
        static_cast<utf8proc_option_t>(UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT | UTF8PROC_STABLE));
    if (n < 0 || !out) {
        if (out) std::free(out);
        return s;  // conservative fallback
    }
    std::string r(reinterpret_cast<char*>(out), static_cast<size_t>(n));
    std::free(out);
    return r;
}

bool is_emoji(char32_t c) {
    return (c >= 0x1F600 && c <= 0x1F64F) || (c >= 0x1F300 && c <= 0x1F5FF) ||
           (c >= 0x1F680 && c <= 0x1F6FF) || (c >= 0x1F700 && c <= 0x1F77F) ||
           (c >= 0x1F780 && c <= 0x1F7FF) || (c >= 0x1F800 && c <= 0x1F8FF) ||
           (c >= 0x1F900 && c <= 0x1F9FF) || (c >= 0x1FA00 && c <= 0x1FA6F) ||
           (c >= 0x1FA70 && c <= 0x1FAFF) || (c >= 0x2600  && c <= 0x26FF)  ||
           (c >= 0x2700  && c <= 0x27BF)  || (c >= 0x1F1E6 && c <= 0x1F1FF);
}

// helper.py::_char_repl + the `[♥☆♡©\\]` strip, on codepoints. `drop`==true → delete the char.
char32_t char_repl(char32_t c, bool& drop) {
    drop = false;
    switch (c) {
        case 0x2013: case 0x2011: case 0x2014: return '-';   // – ‑ —
        case '_':                              return ' ';
        case 0x201C: case 0x201D:              return '"';   // “ ”
        case 0x2018: case 0x2019:              return '\'';  // ‘ ’
        case 0x00B4: case '`':                 return '\'';  // ´ `
        case '[': case ']': case '|': case '/': case '#': return ' ';
        case 0x2192: case 0x2190:              return ' ';   // → ←
        case 0x2665: case 0x2606: case 0x2661: case 0x00A9: case '\\':
            drop = true; return 0;                            // ♥ ☆ ♡ © backslash
        default: return c;
    }
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

bool is_ws(char32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool ends_with_terminal(const std::vector<char32_t>& v) {
    if (v.empty()) return false;
    // helper.py: [.!?;:,'"')\]}…。」』】〉》›»]$
    static const std::u32string kTerm =
        U".!?;:,'\")]}…。」』】〉》›»";
    return kTerm.find(v.back()) != std::u32string::npos;
}

}  // namespace

SupertonicTokenizer::SupertonicTokenizer(const std::string& unicode_indexer_path,
                                         const std::string& /*tts_json_path*/) {
    // unicode_indexer.json is a flat array of 65536 ints (codepoint → id, -1 if unsupported).
    // speech-core's util/json.h has no array parser, so walk it with the primitives.
    const std::string text = json::read_file(unicode_indexer_path);
    if (text.empty())
        throw std::runtime_error("Supertonic: cannot read " + unicode_indexer_path);
    size_t i = 0;
    json::skip_ws(text, i);
    if (i >= text.size() || text[i] != '[')
        throw std::runtime_error("Supertonic: unicode_indexer.json must be a flat JSON array");
    ++i;
    indexer_.reserve(65536);
    while (i < text.size()) {
        json::skip_ws(text, i);
        if (text[i] == ']') { ++i; break; }
        if (text[i] == ',') { ++i; continue; }
        const std::string v = json::parse_value_raw(text, i);
        if (!v.empty())
            indexer_.push_back(static_cast<int32_t>(std::strtol(v.c_str(), nullptr, 10)));
    }
    if (indexer_.empty())
        throw std::runtime_error("Supertonic: unicode_indexer.json parsed empty");
}

bool SupertonicTokenizer::supports(const std::string& lang) const {
    return available_langs().count(lang) != 0;
}

int32_t SupertonicTokenizer::lookup(uint32_t cp) const {
    if (cp < indexer_.size()) return indexer_[cp];
    return kUnknownId;
}

std::string SupertonicTokenizer::preprocess(const std::string& text, const std::string& lang) const {
    // 1) NFKD.
    std::string s = nfkd(text);

    // 2) emoji removal + char-level replacement/strip (codepoint pass).
    {
        std::vector<char32_t> in = utf8_to_u32(s);
        std::vector<char32_t> out;
        out.reserve(in.size());
        for (char32_t c : in) {
            if (is_emoji(c)) continue;
            bool drop = false;
            char32_t r = char_repl(c, drop);
            if (drop) continue;
            out.push_back(r);
        }
        s = u32_to_utf8(out);
    }

    // 3) expression replacements (helper.py::_expr_repl).
    replace_all(s, "@", " at ");
    replace_all(s, "e.g.,", "for example, ");
    replace_all(s, "i.e.,", "that is, ");

    // 4) drop a space before punctuation: " ," -> ",", " ." -> ".", ...
    for (const char* p : {",", ".", "!", "?", ";", ":", "'"}) {
        replace_all(s, std::string(" ") + p, p);
    }

    // 5) collapse repeated quotes.
    while (s.find("\"\"") != std::string::npos) replace_all(s, "\"\"", "\"");
    while (s.find("''")   != std::string::npos) replace_all(s, "''", "'");
    while (s.find("``")   != std::string::npos) replace_all(s, "``", "`");

    // 6) collapse whitespace + trim, then 7) ensure terminal punctuation.
    {
        std::vector<char32_t> in = utf8_to_u32(s);
        std::vector<char32_t> out;
        out.reserve(in.size());
        bool prev_ws = false;
        for (char32_t c : in) {
            if (is_ws(c)) {
                if (!prev_ws) out.push_back(' ');
                prev_ws = true;
            } else {
                out.push_back(c);
                prev_ws = false;
            }
        }
        while (!out.empty() && out.front() == ' ') out.erase(out.begin());
        while (!out.empty() && out.back() == ' ')  out.pop_back();
        if (!ends_with_terminal(out)) out.push_back('.');
        s = u32_to_utf8(out);
    }

    // 8) validate language + wrap.
    if (!available_langs().count(lang))
        throw std::invalid_argument("Supertonic: unsupported language '" + lang + "'");
    return "<" + lang + ">" + s + "</" + lang + ">";
}

SupertonicTokenizer::Tokens
SupertonicTokenizer::process(const std::string& text, const std::string& lang, int text_t) const {
    const std::string wrapped = preprocess(text, lang);
    const std::vector<char32_t> cps = utf8_to_u32(wrapped);

    Tokens t;
    t.ids.assign(static_cast<size_t>(text_t), 0);
    t.mask.assign(static_cast<size_t>(text_t), 0.0f);
    const int n = std::min<int>(static_cast<int>(cps.size()), text_t);
    for (int i = 0; i < n; ++i) {
        t.ids[i]  = lookup(static_cast<uint32_t>(cps[i]));
        t.mask[i] = 1.0f;
    }
    return t;
}

std::vector<std::string>
SupertonicTokenizer::chunk(const std::string& text, const std::string& lang, int max_codepoints) const {
    // Cap so the wrapped, tokenized form fits the exported fixed text length (max_text_tokens_).
    // tag overhead = len("<lang>") + len("</lang>") = 2*lang + 5.
    int cap = max_text_tokens_ - (2 * static_cast<int>(lang.size()) + 5) - 1;
    if (max_codepoints > 0 && max_codepoints < cap) cap = max_codepoints;  // fixed-window duration cap
    if (cap < 8) cap = 8;

    const std::vector<char32_t> cps = utf8_to_u32(text);

    // Sentence-ish split at terminal punctuation followed by whitespace (faithful subset of
    // helper.py::_chunk_text; the abbreviation-guard regex there is a refinement — boundary
    // differences only shift where the 0.3 s inter-chunk silence lands, not parity).
    static const std::u32string kTerm = U".!?…。！？";
    std::vector<std::vector<char32_t>> sentences;
    std::vector<char32_t> cur;
    for (size_t i = 0; i < cps.size(); ++i) {
        cur.push_back(cps[i]);
        const bool term    = kTerm.find(cps[i]) != std::u32string::npos;
        const bool nextws  = (i + 1 < cps.size()) && is_ws(cps[i + 1]);
        if (term && nextws) { sentences.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) sentences.push_back(cur);

    std::vector<std::string> out;
    std::vector<char32_t> chunk_cps;
    auto flush = [&]() {
        size_t a = 0, b = chunk_cps.size();
        while (a < b && chunk_cps[a] == ' ') ++a;
        while (b > a && chunk_cps[b - 1] == ' ') --b;
        if (b > a) out.push_back(u32_to_utf8(std::vector<char32_t>(chunk_cps.begin() + a,
                                                                   chunk_cps.begin() + b)));
        chunk_cps.clear();
    };
    auto fits = [&](int extra) {
        return static_cast<int>(chunk_cps.size()) + (chunk_cps.empty() ? 0 : 1) + extra <= cap;
    };
    auto append_unit = [&](const std::vector<char32_t>& u) {
        if (!chunk_cps.empty()) chunk_cps.push_back(' ');
        chunk_cps.insert(chunk_cps.end(), u.begin(), u.end());
    };

    for (auto& sent : sentences) {
        if (static_cast<int>(sent.size()) <= cap) {
            if (!fits(static_cast<int>(sent.size()))) flush();
            append_unit(sent);
            continue;
        }
        // Oversize sentence: hard-split on word boundaries (and pathologically long tokens).
        flush();
        std::vector<char32_t> word;
        auto push_word = [&]() {
            if (word.empty()) return;
            if (!fits(static_cast<int>(word.size()))) flush();
            append_unit(word);
            word.clear();
        };
        for (char32_t c : sent) {
            if (is_ws(c)) { push_word(); continue; }
            word.push_back(c);
            if (static_cast<int>(word.size()) >= cap) push_word();
        }
        push_word();
    }
    flush();
    if (out.empty()) out.emplace_back();  // degenerate input → one empty chunk
    return out;
}

}  // namespace speech_core
