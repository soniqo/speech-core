#include "speech_core/models/voxcpm2_tokenizer.h"

#include "speech_core/util/json.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace speech_core {

// SentencePiece-style space marker: U+2581 LOWER ONE EIGHTH BLOCK ("▁").
// Spaces are replaced with this single Unicode character before BPE so the
// tokenizer can recover word boundaries during decode.
static constexpr const char* kSpaceMarker = "\xE2\x96\x81";

// ---------------------------------------------------------------------------
// JSON helpers — minimal, scoped to this file because util/json.h doesn't
// expose nested-object navigation. tokenizer.json is too large for the flat
// parser and we don't want to add a JSON dependency just for this file.
// ---------------------------------------------------------------------------

namespace {

// Find the value substring for "key" within an object `s[i..end]`. Returns
// the index of the first char of the value (past the colon and whitespace),
// or std::string::npos if not found. Assumes `i` points just inside `{`.
size_t find_member(const std::string& s, size_t i, size_t end, const std::string& key) {
    while (i < end) {
        ::json::skip_ws(s, i);
        if (i >= end || s[i] == '}') return std::string::npos;
        if (s[i] == ',') { ++i; continue; }

        size_t key_start = i;
        std::string k = ::json::parse_string(s, i);
        (void)key_start;
        ::json::skip_ws(s, i);
        if (i < end && s[i] == ':') ++i;
        ::json::skip_ws(s, i);
        if (k == key) return i;
        ::json::skip_value(s, i);
    }
    return std::string::npos;
}

// Parse an integer token literal (no exponent, no fraction) at position `i`.
int parse_int(const std::string& s, size_t& i) {
    bool neg = false;
    if (i < s.size() && s[i] == '-') { neg = true; ++i; }
    long long val = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        val = val * 10 + (s[i] - '0');
        ++i;
    }
    return static_cast<int>(neg ? -val : val);
}

// Parse a bool literal ("true" / "false") at position `i`.
bool parse_bool(const std::string& s, size_t& i) {
    if (i + 3 < s.size() && std::strncmp(&s[i], "true", 4) == 0) { i += 4; return true; }
    if (i + 4 < s.size() && std::strncmp(&s[i], "false", 5) == 0) { i += 5; return false; }
    return false;
}

// Decompose a UTF-8 string into a vector of "characters" (each a 1–4 byte
// substring representing one Unicode codepoint). Invalid sequences pass
// through one byte at a time.
std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if      ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        // Clamp in case the buffer is truncated mid-codepoint.
        if (i + len > s.size()) len = s.size() - i;
        out.emplace_back(s.substr(i, len));
        i += len;
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructor: parse tokenizer.json
// ---------------------------------------------------------------------------

VoxCPM2Tokenizer::VoxCPM2Tokenizer(const std::string& path) {
    std::string text = ::json::read_file(path);
    if (text.empty()) {
        throw std::runtime_error("VoxCPM2Tokenizer: cannot read " + path);
    }

    // Find the root object.
    size_t root = 0;
    ::json::skip_ws(text, root);
    if (root >= text.size() || text[root] != '{') {
        throw std::runtime_error("VoxCPM2Tokenizer: tokenizer.json is not a JSON object");
    }
    ++root;
    const size_t end = text.size();

    // --- added_tokens: array of {id, content, special} ---
    size_t at_pos = find_member(text, root, end, "added_tokens");
    if (at_pos != std::string::npos && at_pos < end && text[at_pos] == '[') {
        size_t p = at_pos + 1;
        while (p < end) {
            ::json::skip_ws(text, p);
            if (p >= end || text[p] == ']') break;
            if (text[p] == ',') { ++p; continue; }
            if (text[p] != '{') { ++p; continue; }

            size_t obj_start = p + 1;
            size_t obj_end_scan = p;
            ::json::skip_value(text, obj_end_scan);
            const size_t obj_end = obj_end_scan;

            int  id = -1;
            std::string content;
            bool special = false;
            for (auto key : {"id", "content", "special"}) {
                size_t v = find_member(text, obj_start, obj_end, key);
                if (v == std::string::npos) continue;
                if (std::strcmp(key, "id") == 0) {
                    id = parse_int(text, v);
                } else if (std::strcmp(key, "content") == 0) {
                    content = ::json::parse_string(text, v);
                } else if (std::strcmp(key, "special") == 0) {
                    special = parse_bool(text, v);
                }
            }
            if (id >= 0 && !content.empty()) {
                if (static_cast<int>(id_to_token_.size()) <= id) id_to_token_.resize(id + 1);
                id_to_token_[id] = content;
                vocab_[content] = id;
                if (special) {
                    special_ids_.insert(id);
                    if      (content == "<s>")   bos_id_ = id;
                    else if (content == "</s>")  eos_id_ = id;
                    else if (content == "<unk>") unk_id_ = id;
                }
            }
            p = obj_end;
        }
    }

    // --- model.* ---
    size_t model_pos = find_member(text, root, end, "model");
    if (model_pos == std::string::npos || text[model_pos] != '{') {
        throw std::runtime_error("VoxCPM2Tokenizer: missing model object");
    }
    const size_t model_inner = model_pos + 1;
    size_t model_end_scan = model_pos;
    ::json::skip_value(text, model_end_scan);
    const size_t model_end = model_end_scan;

    // byte_fallback
    size_t bf = find_member(text, model_inner, model_end, "byte_fallback");
    if (bf != std::string::npos) byte_fallback_ = parse_bool(text, bf);

    // vocab: {"token": id, ...}
    size_t vocab_pos = find_member(text, model_inner, model_end, "vocab");
    if (vocab_pos == std::string::npos || text[vocab_pos] != '{') {
        throw std::runtime_error("VoxCPM2Tokenizer: missing model.vocab");
    }
    size_t v = vocab_pos + 1;
    while (v < model_end) {
        ::json::skip_ws(text, v);
        if (v >= model_end || text[v] == '}') break;
        if (text[v] == ',') { ++v; continue; }

        std::string tok = ::json::parse_string(text, v);
        ::json::skip_ws(text, v);
        if (v < model_end && text[v] == ':') ++v;
        ::json::skip_ws(text, v);
        int id = parse_int(text, v);
        if (static_cast<int>(id_to_token_.size()) <= id) id_to_token_.resize(id + 1);
        // added_tokens parsed above take precedence (already populated); for any
        // overlap we keep the existing mapping. For non-added tokens this just
        // assigns the standard vocab entry.
        if (vocab_.find(tok) == vocab_.end()) {
            vocab_[tok] = id;
            id_to_token_[id] = tok;
        }
    }

    // merges: ["left right", ...]  (lower index = higher merge priority)
    size_t merges_pos = find_member(text, model_inner, model_end, "merges");
    if (merges_pos == std::string::npos || text[merges_pos] != '[') {
        throw std::runtime_error("VoxCPM2Tokenizer: missing model.merges");
    }
    size_t m = merges_pos + 1;
    int merge_idx = 0;
    while (m < model_end) {
        ::json::skip_ws(text, m);
        if (m >= model_end || text[m] == ']') break;
        if (text[m] == ',') { ++m; continue; }
        if (text[m] != '"') { ++m; continue; }
        std::string entry = ::json::parse_string(text, m);
        merge_rank_[entry] = merge_idx++;
    }

    // Cache byte-fallback token IDs for 0x00..0xFF.
    if (byte_fallback_) {
        byte_token_ids_.assign(256, -1);
        char buf[8];
        for (int b = 0; b < 256; ++b) {
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            auto it = vocab_.find(buf);
            if (it != vocab_.end()) byte_token_ids_[b] = it->second;
        }
    }
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

int VoxCPM2Tokenizer::token_id(const std::string& token) const {
    auto it = vocab_.find(token);
    return it != vocab_.end() ? it->second : -1;
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

std::vector<int> VoxCPM2Tokenizer::encode(const std::string& text) const {
    // Normalize: prepend ▁ and replace every space with ▁.
    std::string normalized = kSpaceMarker;
    for (char c : text) {
        if (c == ' ') normalized += kSpaceMarker;
        else          normalized += c;
    }
    // Empty input still receives the BOS prefix but no content tokens.
    std::vector<int> out;
    out.push_back(bos_id_);
    if (text.empty()) return out;

    auto ids = bpe(normalized);
    out.insert(out.end(), ids.begin(), ids.end());
    return out;
}

std::vector<int> VoxCPM2Tokenizer::bpe(const std::string& word) const {
    // Initial token list: one entry per Unicode codepoint. Codepoints not in
    // the vocab are split into their UTF-8 bytes (byte fallback).
    std::vector<std::string> tokens;
    for (const auto& ch : utf8_chars(word)) {
        if (vocab_.find(ch) != vocab_.end()) {
            tokens.push_back(ch);
        } else if (byte_fallback_) {
            for (unsigned char b : ch) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
                tokens.emplace_back(buf);
            }
        } else {
            tokens.push_back(ch);  // will resolve to <unk> below if missing
        }
    }

    // Greedy lowest-rank-pair merging. O(N^2) but each step shrinks the list
    // and inputs are short prompts — fine for TTS-scale text.
    while (tokens.size() >= 2) {
        int  best_rank = std::numeric_limits<int>::max();
        long best_idx  = -1;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            std::string key = tokens[i] + ' ' + tokens[i + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx  = static_cast<long>(i);
            }
        }
        if (best_idx < 0) break;
        tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1];
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    // Resolve each token to its ID; missing → <unk>.
    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (const auto& t : tokens) {
        auto it = vocab_.find(t);
        ids.push_back(it != vocab_.end() ? it->second : unk_id_);
    }
    return ids;
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

std::string VoxCPM2Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    // Buffer for collapsing consecutive <0xXX> tokens back into raw bytes
    // before they hit the rest of the decoder pipeline.
    std::vector<unsigned char> byte_buf;
    auto flush_bytes = [&]() {
        if (byte_buf.empty()) return;
        out.append(reinterpret_cast<const char*>(byte_buf.data()), byte_buf.size());
        byte_buf.clear();
    };

    for (int id : ids) {
        if (special_ids_.count(id)) { flush_bytes(); continue; }
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
        const std::string& tok = id_to_token_[id];
        if (tok.empty()) continue;

        // Detect byte-fallback tokens: literal "<0xXX>" (6 chars).
        if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok.back() == '>') {
            unsigned int b = 0;
            if (std::sscanf(tok.c_str() + 3, "%2X", &b) == 1) {
                byte_buf.push_back(static_cast<unsigned char>(b));
                continue;
            }
        }
        flush_bytes();

        // Replace ▁ with space inside the token.
        for (size_t j = 0; j < tok.size(); ) {
            if (j + 3 <= tok.size() && std::memcmp(&tok[j], kSpaceMarker, 3) == 0) {
                out += ' ';
                j += 3;
            } else {
                out += tok[j++];
            }
        }
    }
    flush_bytes();

    // Strip the single leading space introduced by the normalizer.
    if (!out.empty() && out.front() == ' ') out.erase(out.begin());
    return out;
}

}  // namespace speech_core
