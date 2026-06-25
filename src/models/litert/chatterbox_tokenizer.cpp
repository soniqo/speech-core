#include "speech_core/models/chatterbox_tokenizer.h"

#include "speech_core/util/json.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace speech_core {

namespace {

// Minimal JSON object navigation (util/json.h is a flat parser). Same helpers
// as voxcpm2_tokenizer.cpp.
size_t find_member(const std::string& s, size_t i, size_t end, const std::string& key) {
    while (i < end) {
        ::json::skip_ws(s, i);
        if (i >= end || s[i] == '}') return std::string::npos;
        if (s[i] == ',') { ++i; continue; }
        std::string k = ::json::parse_string(s, i);
        ::json::skip_ws(s, i);
        if (i < end && s[i] == ':') ++i;
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
std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> out; out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        size_t len = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        if (i + len > s.size()) len = s.size() - i;
        out.emplace_back(s.substr(i, len)); i += len;
    }
    return out;
}
// A utf8 char is a Whitespace-pretokenizer "word" char (\w) if it is ASCII
// alnum/underscore, or any non-ASCII byte sequence (treated as a letter). ASCII
// punctuation/symbols are non-word. (Full Unicode \w would need property tables;
// this is exact for ASCII + correct for the Latin/Arabic letter case.)
bool is_word_char(const std::string& ch) {
    if (ch.size() == 1) { unsigned char c = (unsigned char)ch[0];
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
    return true;  // non-ASCII -> letter
}
// ASCII-only lowercase (Unicode case folding deferred; Arabic is caseless and
// English is ASCII, so this is exact for those; accented Latin needs NFKD+fold).
std::string ascii_lower(const std::string& s) {
    std::string o = s;
    for (char& c : o) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return o;
}

// punc_norm port (ChatterboxMultilingualTTS.punc_norm) — the subsequent lower()
// makes the leading capitalisation moot, so we skip that step.
std::string punc_norm(const std::string& in) {
    if (in.empty()) return "You need to add some text for me to talk.";
    // collapse runs of ASCII whitespace to single spaces
    std::string t; bool sp = false;
    for (char c : in) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { sp = true; }
        else { if (sp && !t.empty()) t += ' '; sp = false; t += c; }
    }
    auto rep = [&](const std::string& a, const std::string& b) {
        size_t p = 0; while ((p = t.find(a, p)) != std::string::npos) { t.replace(p, a.size(), b); p += b.size(); }
    };
    rep("...", ", "); rep("\xE2\x80\xA6", ", ");  // …
    rep(":", ","); rep(" - ", ", "); rep(";", ", ");
    rep("\xE2\x80\x94", "-"); rep("\xE2\x80\x93", "-");  // — –
    rep(" ,", ",");
    rep("\xE2\x80\x9C", "\""); rep("\xE2\x80\x9D", "\"");  // “ ”
    rep("\xE2\x80\x98", "'"); rep("\xE2\x80\x99", "'");    // ‘ ’
    while (!t.empty() && t.back() == ' ') t.pop_back();
    static const char* enders[] = {".", "!", "?", "-", ",",
        "\xE3\x80\x81", "\xEF\xBC\x8C", "\xE3\x80\x82", "\xEF\xBC\x9F", "\xEF\xBC\x81"};
    bool ok = false;
    for (auto e : enders) { size_t n = std::strlen(e); if (t.size() >= n && t.compare(t.size() - n, n, e) == 0) { ok = true; break; } }
    if (!ok) t += ".";
    return t;
}

// Algorithmic Hangul -> Jamo (korean_normalize).
std::string hangul_to_jamo(const std::string& s) {
    std::string out;
    for (const auto& ch : utf8_chars(s)) {
        // decode codepoint
        unsigned cp = 0; unsigned char c0 = (unsigned char)ch[0];
        if (ch.size() == 1) cp = c0;
        else if (ch.size() == 2) cp = ((c0 & 0x1F) << 6) | ((unsigned char)ch[1] & 0x3F);
        else if (ch.size() == 3) cp = ((c0 & 0x0F) << 12) | (((unsigned char)ch[1] & 0x3F) << 6) | ((unsigned char)ch[2] & 0x3F);
        else { out += ch; continue; }
        if (cp < 0xAC00 || cp > 0xD7A3) { out += ch; continue; }
        unsigned b = cp - 0xAC00;
        unsigned L = 0x1100 + b / (21 * 28), Vv = 0x1161 + (b % (21 * 28)) / 28, Tt = b % 28;
        auto enc = [&](unsigned u) { out += char(0xE0 | (u >> 12)); out += char(0x80 | ((u >> 6) & 0x3F)); out += char(0x80 | (u & 0x3F)); };
        enc(L); enc(Vv); if (Tt) enc(0x11A7 + Tt);
    }
    return out;
}

}  // namespace

ChatterboxTokenizer::ChatterboxTokenizer(const std::string& tokenizer_json, const std::string& cangjie_json) {
    std::string text = ::json::read_file(tokenizer_json);
    if (text.empty()) throw std::runtime_error("ChatterboxTokenizer: cannot read " + tokenizer_json);
    size_t root = 0; ::json::skip_ws(text, root);
    if (root >= text.size() || text[root] != '{') throw std::runtime_error("ChatterboxTokenizer: not a JSON object");
    ++root; const size_t end = text.size();

    // added_tokens: [{id, content}]
    size_t at = find_member(text, root, end, "added_tokens");
    if (at != std::string::npos && text[at] == '[') {
        size_t p = at + 1;
        while (p < end) {
            ::json::skip_ws(text, p);
            if (p >= end || text[p] == ']') break;
            if (text[p] == ',') { ++p; continue; }
            if (text[p] != '{') { ++p; continue; }
            size_t obj_start = p + 1, scan = p; ::json::skip_value(text, scan); size_t obj_end = scan;
            int id = -1; std::string content;
            for (auto key : {"id", "content"}) {
                size_t v = find_member(text, obj_start, obj_end, key);
                if (v == std::string::npos) continue;
                if (std::strcmp(key, "id") == 0) id = parse_int(text, v);
                else content = ::json::parse_string(text, v);
            }
            if (id >= 0 && !content.empty()) {
                if ((int)id_to_token_.size() <= id) id_to_token_.resize(id + 1);
                id_to_token_[id] = content; vocab_[content] = id;
                added_.push_back(content); added_id_[content] = id;
                if (content == "[UNK]") unk_id_ = id;
            }
            p = obj_end;
        }
    }
    // model.vocab + model.merges
    size_t mp = find_member(text, root, end, "model");
    if (mp == std::string::npos || text[mp] != '{') throw std::runtime_error("ChatterboxTokenizer: missing model");
    size_t minner = mp + 1, mscan = mp; ::json::skip_value(text, mscan); size_t mend = mscan;
    size_t vp = find_member(text, minner, mend, "vocab");
    if (vp == std::string::npos || text[vp] != '{') throw std::runtime_error("ChatterboxTokenizer: missing vocab");
    size_t v = vp + 1;
    while (v < mend) {
        ::json::skip_ws(text, v);
        if (v >= mend || text[v] == '}') break;
        if (text[v] == ',') { ++v; continue; }
        std::string tok = ::json::parse_string(text, v);
        ::json::skip_ws(text, v); if (v < mend && text[v] == ':') ++v; ::json::skip_ws(text, v);
        int id = parse_int(text, v);
        if ((int)id_to_token_.size() <= id) id_to_token_.resize(id + 1);
        if (vocab_.find(tok) == vocab_.end()) { vocab_[tok] = id; id_to_token_[id] = tok; }
    }
    size_t gp = find_member(text, minner, mend, "merges");
    if (gp != std::string::npos && text[gp] == '[') {
        size_t m = gp + 1; int idx = 0;
        while (m < mend) {
            ::json::skip_ws(text, m);
            if (m >= mend || text[m] == ']') break;
            if (text[m] == ',') { ++m; continue; }
            if (text[m] != '"') { ++m; continue; }
            std::string e = ::json::parse_string(text, m);
            merge_rank_[e] = idx++;
        }
    }
    // longest-first so greedy added-token matching prefers the longest token
    std::sort(added_.begin(), added_.end(), [](const std::string& a, const std::string& b) { return a.size() > b.size(); });

    // Cangjie map: array of "word\tcode...".
    if (!cangjie_json.empty()) {
        std::string cj = ::json::read_file(cangjie_json);
        size_t i = 0;
        while (i < cj.size()) {
            if (cj[i] != '"') { ++i; continue; }
            std::string entry = ::json::parse_string(cj, i);
            size_t tab = entry.find('\t');
            if (tab != std::string::npos) {
                std::string word = entry.substr(0, tab), rest = entry.substr(tab + 1);
                size_t tab2 = rest.find('\t'); std::string code = tab2 == std::string::npos ? rest : rest.substr(0, tab2);
                if (cangjie_.find(word) == cangjie_.end()) cangjie_[word] = code;
            }
        }
    }
}

int ChatterboxTokenizer::token_id(const std::string& t) const {
    auto it = vocab_.find(t); return it != vocab_.end() ? it->second : -1;
}

std::vector<int> ChatterboxTokenizer::bpe_word(const std::string& word) const {
    std::vector<std::string> toks = utf8_chars(word);
    while (toks.size() >= 2) {
        int best = std::numeric_limits<int>::max(); long bi = -1;
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            auto it = merge_rank_.find(toks[i] + ' ' + toks[i + 1]);
            if (it != merge_rank_.end() && it->second < best) { best = it->second; bi = (long)i; }
        }
        if (bi < 0) break;
        toks[bi] += toks[bi + 1]; toks.erase(toks.begin() + bi + 1);
    }
    std::vector<int> ids; ids.reserve(toks.size());
    for (auto& t : toks) { auto it = vocab_.find(t); ids.push_back(it != vocab_.end() ? it->second : unk_id_); }
    return ids;
}

// Whitespace pretokenize a gap (no spaces, no added tokens) into \w+ / [^\w\s]+
// runs, BPE each.
std::vector<int> ChatterboxTokenizer::tokenize_segment(const std::string& seg) const {
    std::vector<int> out;
    auto chars = utf8_chars(seg);
    size_t i = 0;
    while (i < chars.size()) {
        bool w = is_word_char(chars[i]);
        std::string run;
        while (i < chars.size() && is_word_char(chars[i]) == w) { run += chars[i]; ++i; }
        auto ids = bpe_word(run);
        out.insert(out.end(), ids.begin(), ids.end());
    }
    return out;
}

std::vector<int> ChatterboxTokenizer::encode(const std::string& text_in, const std::string& lang) const {
    std::string t = punc_norm(text_in);
    t = ascii_lower(t);                      // (NFKD deferred)
    if (lang == "ko") t = hangul_to_jamo(t);
    else if (lang == "zh" && !cangjie_.empty()) {
        std::string o;
        for (const auto& ch : utf8_chars(t)) {
            auto it = cangjie_.find(ch);
            if (it != cangjie_.end()) { for (char c : it->second) { o += "[cj_"; o += c; o += "]"; } o += "[cj_.]"; }
            else o += ch;
        }
        t = o;
    }
    // ja/he/ru: pass-through (matches the reference env without kakasi/dicta/stress)
    if (!lang.empty()) t = "[" + ascii_lower(lang) + "]" + t;
    // replace spaces with the [SPACE] added token
    { std::string o; for (const auto& ch : utf8_chars(t)) o += (ch == " " ? "[SPACE]" : ch); t = o; }

    // split on added/special tokens (greedy longest-first), BPE the gaps
    std::vector<int> out;
    std::string gap;
    size_t i = 0;
    while (i < t.size()) {
        const std::string* hit = nullptr;
        for (const auto& a : added_) {
            if (i + a.size() <= t.size() && t.compare(i, a.size(), a) == 0) { hit = &a; break; }
        }
        if (hit) {
            if (!gap.empty()) { auto g = tokenize_segment(gap); out.insert(out.end(), g.begin(), g.end()); gap.clear(); }
            out.push_back(added_id_.at(*hit));
            i += hit->size();
        } else {
            // advance one utf8 char into the gap
            unsigned char c = (unsigned char)t[i];
            size_t len = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
            if (i + len > t.size()) len = t.size() - i;
            gap += t.substr(i, len); i += len;
        }
    }
    if (!gap.empty()) { auto g = tokenize_segment(gap); out.insert(out.end(), g.begin(), g.end()); }
    return out;
}

}  // namespace speech_core
