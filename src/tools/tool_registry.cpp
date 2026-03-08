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

// Skip whitespace
size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                               s[pos] == '\n' || s[pos] == '\r'))
        pos++;
    return pos;
}

// Parse a JSON string value (expects pos at opening quote)
// Returns the string content and advances pos past closing quote
bool parse_string(const std::string& s, size_t& pos, std::string& out) {
    if (pos >= s.size() || s[pos] != '"') return false;
    pos++;  // skip opening quote
    out.clear();
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            pos++;
            switch (s[pos]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                default: out += s[pos]; break;
            }
        } else {
            out += s[pos];
        }
        pos++;
    }
    if (pos >= s.size()) return false;
    pos++;  // skip closing quote
    return true;
}

// Parse a JSON integer
bool parse_int(const std::string& s, size_t& pos, int& out) {
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-') pos++;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
    if (pos == start) return false;
    out = std::stoi(s.substr(start, pos - start));
    return true;
}

// Parse a JSON string array
bool parse_string_array(const std::string& s, size_t& pos,
                        std::vector<std::string>& out)
{
    if (pos >= s.size() || s[pos] != '[') return false;
    pos++;  // skip [
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

// Parse a single tool object
bool parse_tool(const std::string& s, size_t& pos, ToolDefinition& tool) {
    if (pos >= s.size() || s[pos] != '{') return false;
    pos++;  // skip {

    while (pos < s.size()) {
        pos = skip_ws(s, pos);
        if (s[pos] == '}') { pos++; return true; }
        if (s[pos] == ',') { pos++; continue; }

        // Parse key
        std::string key;
        if (!parse_string(s, pos, key)) return false;

        pos = skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ':') return false;
        pos++;  // skip :
        pos = skip_ws(s, pos);

        // Parse value based on key
        if (key == "name") {
            if (!parse_string(s, pos, tool.name)) return false;
        } else if (key == "description") {
            if (!parse_string(s, pos, tool.description)) return false;
        } else if (key == "command") {
            if (!parse_string(s, pos, tool.command)) return false;
        } else if (key == "triggers") {
            if (!parse_string_array(s, pos, tool.triggers)) return false;
        } else if (key == "timeout") {
            if (!parse_int(s, pos, tool.timeout)) return false;
        } else if (key == "cooldown") {
            if (!parse_int(s, pos, tool.cooldown)) return false;
        } else {
            // Skip unknown value (string or number)
            if (pos < s.size() && s[pos] == '"') {
                std::string dummy;
                if (!parse_string(s, pos, dummy)) return false;
            } else {
                int dummy;
                if (!parse_int(s, pos, dummy)) return false;
            }
        }
    }
    return false;
}

}  // namespace

int ToolRegistry::load_json(const std::string& json) {
    size_t pos = skip_ws(json, 0);
    if (pos >= json.size() || json[pos] != '[') return -1;
    pos++;  // skip [

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
    return -1;  // missing ]
}

}  // namespace speech_core
