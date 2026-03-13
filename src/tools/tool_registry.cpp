#include "speech_core/tools/tool_registry.h"

#include <algorithm>

namespace speech_core {

void ToolRegistry::add(ToolDefinition tool) {
    tools_.push_back(std::move(tool));
}

const ToolDefinition* ToolRegistry::find(const std::string& name) const {
    for (const auto& t : tools_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Minimal JSON parser for tool definitions
// ---------------------------------------------------------------------------

namespace {

size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                               s[pos] == '\n' || s[pos] == '\r'))
        pos++;
    return pos;
}

bool parse_string(const std::string& s, size_t& pos, std::string& out) {
    if (pos >= s.size() || s[pos] != '"') return false;
    pos++;
    out.clear();
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            pos++;
            switch (s[pos]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    // Parse \uXXXX — decode BMP codepoints to UTF-8
                    if (pos + 4 >= s.size()) return false;
                    unsigned cp = 0;
                    for (int i = 1; i <= 4; i++) {
                        char c = s[pos + i];
                        cp <<= 4;
                        if (c >= '0' && c <= '9')      cp |= (c - '0');
                        else if (c >= 'a' && c <= 'f') cp |= (c - 'a' + 10);
                        else if (c >= 'A' && c <= 'F') cp |= (c - 'A' + 10);
                        else return false;
                    }
                    pos += 4;
                    if (cp < 0x80) {
                        out += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        out += static_cast<char>(0xC0 | (cp >> 6));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (cp >> 12));
                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: out += s[pos]; break;
            }
        } else {
            out += s[pos];
        }
        pos++;
    }
    if (pos >= s.size()) return false;
    pos++;
    return true;
}

bool parse_number(const std::string& s, size_t& pos, int& out) {
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-') pos++;
    if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') {
        pos = start;
        return false;
    }
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
    // Skip fractional part (e.g. 5.0 → 5)
    if (pos < s.size() && s[pos] == '.') {
        pos++;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
    }
    // Skip exponent (e.g. 1e3)
    if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
        pos++;
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) pos++;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
    }
    // Convert integer part only (truncate)
    try {
        out = std::stoi(s.substr(start, pos - start));
    } catch (...) {
        out = 0;
    }
    return true;
}

// Forward declaration
bool skip_value(const std::string& s, size_t& pos);

bool skip_array(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '[') return false;
    pos++;
    pos = skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') { pos++; return true; }
    while (pos < s.size()) {
        if (!skip_value(s, pos)) return false;
        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ']') { pos++; return true; }
        if (pos < s.size() && s[pos] == ',') { pos++; continue; }
        return false;
    }
    return false;
}

bool skip_object(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '{') return false;
    pos++;
    while (pos < s.size()) {
        pos = skip_ws(s, pos);
        if (s[pos] == '}') { pos++; return true; }
        if (s[pos] == ',') { pos++; continue; }
        std::string key;
        if (!parse_string(s, pos, key)) return false;
        pos = skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ':') return false;
        pos++;
        pos = skip_ws(s, pos);
        if (!skip_value(s, pos)) return false;
    }
    return false;
}

bool skip_value(const std::string& s, size_t& pos) {
    pos = skip_ws(s, pos);
    if (pos >= s.size()) return false;
    switch (s[pos]) {
        case '"': { std::string d; return parse_string(s, pos, d); }
        case '{': return skip_object(s, pos);
        case '[': return skip_array(s, pos);
        case 't':
            if (s.compare(pos, 4, "true") == 0) { pos += 4; return true; }
            return false;
        case 'f':
            if (s.compare(pos, 5, "false") == 0) { pos += 5; return true; }
            return false;
        case 'n':
            if (s.compare(pos, 4, "null") == 0) { pos += 4; return true; }
            return false;
        default: {
            int d;
            return parse_number(s, pos, d);
        }
    }
}

bool parse_string_array(const std::string& s, size_t& pos,
                        std::vector<std::string>& out)
{
    if (pos >= s.size() || s[pos] != '[') return false;
    pos++;
    out.clear();

    pos = skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') { pos++; return true; }

    while (pos < s.size()) {
        pos = skip_ws(s, pos);
        std::string val;
        if (!parse_string(s, pos, val)) return false;
        out.push_back(std::move(val));

        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ']') { pos++; return true; }
        if (pos < s.size() && s[pos] == ',') { pos++; continue; }
        return false;
    }
    return false;
}

bool parse_tool(const std::string& s, size_t& pos, ToolDefinition& tool) {
    if (pos >= s.size() || s[pos] != '{') return false;
    pos++;

    while (pos < s.size()) {
        pos = skip_ws(s, pos);
        if (s[pos] == '}') { pos++; return true; }
        if (s[pos] == ',') { pos++; continue; }

        std::string key;
        if (!parse_string(s, pos, key)) return false;

        pos = skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ':') return false;
        pos++;
        pos = skip_ws(s, pos);

        if (key == "name") {
            if (!parse_string(s, pos, tool.name)) return false;
        } else if (key == "description") {
            if (!parse_string(s, pos, tool.description)) return false;
        } else if (key == "command") {
            if (!parse_string(s, pos, tool.command)) return false;
        } else if (key == "triggers") {
            if (!parse_string_array(s, pos, tool.triggers)) return false;
        } else if (key == "timeout") {
            if (!parse_number(s, pos, tool.timeout)) return false;
        } else if (key == "cooldown") {
            if (!parse_number(s, pos, tool.cooldown)) return false;
        } else {
            if (!skip_value(s, pos)) return false;
        }
    }
    return false;
}

}  // namespace

int ToolRegistry::load_json(const std::string& json) {
    size_t pos = skip_ws(json, 0);
    if (pos >= json.size() || json[pos] != '[') return -1;
    pos++;

    int count = 0;
    while (pos < json.size()) {
        pos = skip_ws(json, pos);
        if (json[pos] == ']') return count;
        if (json[pos] == ',') { pos++; continue; }

        ToolDefinition tool;
        if (!parse_tool(json, pos, tool)) return -1;
        tools_.push_back(std::move(tool));
        count++;
    }
    return -1;
}

}  // namespace speech_core
