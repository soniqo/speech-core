// Force-enable asserts even under Release builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

// IndicMioTokenizer vs golden ids from the reference HF tokenizer
// (tests/data/indic_mio_tokenizer_fixtures.json — Devanagari, emotion tags,
// English contractions, digits, whitespace shapes, and the full chat prompt).
// The tokenizer.json itself is 14 MB and ships with the model bundle, not the
// repo — the test skips unless SPEECH_CORE_INDIC_MIO_BUNDLE points at one.

#include "speech_core/models/indic_mio_tokenizer.h"
#include "speech_core/util/json.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string text;
    std::vector<int> ids;
    bool prompt = false;
};

std::vector<Case> load_cases(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    assert(f && "fixture json missing");
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();

    std::vector<Case> cases;
    size_t i = s.find("\"cases\"");
    assert(i != std::string::npos);
    i = s.find('[', i);
    ++i;
    while (i < s.size()) {
        ::json::skip_ws(s, i);
        if (i >= s.size() || s[i] == ']') break;
        if (s[i] == ',') { ++i; continue; }
        assert(s[i] == '{');
        ++i;
        Case c;
        while (i < s.size() && s[i] != '}') {
            ::json::skip_ws(s, i);
            if (s[i] == ',') { ++i; continue; }
            if (s[i] == '}') break;
            std::string key = ::json::parse_string(s, i);
            ::json::skip_ws(s, i);
            if (s[i] == ':') ++i;
            ::json::skip_ws(s, i);
            if (key == "text") {
                c.text = ::json::parse_string(s, i);
            } else if (key == "ids") {
                assert(s[i] == '[');
                ++i;
                while (i < s.size() && s[i] != ']') {
                    ::json::skip_ws(s, i);
                    if (s[i] == ',') { ++i; continue; }
                    if (s[i] == ']') break;
                    int v = 0;
                    bool neg = s[i] == '-';
                    if (neg) ++i;
                    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                        v = v * 10 + (s[i] - '0');
                        ++i;
                    }
                    c.ids.push_back(neg ? -v : v);
                }
                if (i < s.size()) ++i;  // ']'
            } else if (key == "prompt") {
                c.prompt = s.compare(i, 4, "true") == 0;
                ::json::skip_value(s, i);
            } else {
                ::json::skip_value(s, i);
            }
        }
        if (i < s.size()) ++i;  // '}'
        cases.push_back(std::move(c));
    }
    return cases;
}

// The runner's prompt construction: control ids inserted by id around plain
// encode() segments — must reproduce HF's added-token splitting exactly.
std::vector<int> build_prompt(const speech_core::IndicMioTokenizer& tok,
                              const std::string& text) {
    constexpr int kImStart = 151644, kImEnd = 151645;
    std::vector<int> ids;
    ids.push_back(kImStart);
    for (int v : tok.encode("user\n" + text)) ids.push_back(v);
    ids.push_back(kImEnd);
    for (int v : tok.encode("\n")) ids.push_back(v);
    ids.push_back(kImStart);
    for (int v : tok.encode("assistant\n")) ids.push_back(v);
    return ids;
}

}  // namespace

int main() {
    const char* bundle = std::getenv("SPEECH_CORE_INDIC_MIO_BUNDLE");
    if (!bundle || !*bundle) {
        std::printf("[skip] SPEECH_CORE_INDIC_MIO_BUNDLE not set\n");
        return 0;
    }
    speech_core::IndicMioTokenizer tok(std::string(bundle) + "/tokenizer.json");
    assert(tok.vocab_size() > 150000);

    const auto cases =
        load_cases(std::string(SPEECH_CORE_TEST_DATA_DIR) + "/indic_mio_tokenizer_fixtures.json");
    assert(!cases.empty());

    int failed = 0;
    for (const auto& c : cases) {
        std::vector<int> got;
        if (c.prompt) {
            // The templated case fixes the full chat prompt; extract the user
            // text between the markers the generator used.
            const std::string pre = "<|im_start|>user\n";
            const std::string post = "<|im_end|>\n<|im_start|>assistant\n";
            assert(c.text.size() > pre.size() + post.size());
            const std::string text =
                c.text.substr(pre.size(), c.text.size() - pre.size() - post.size());
            got = build_prompt(tok, text);
        } else {
            got = tok.encode(c.text);
        }
        if (got != c.ids) {
            ++failed;
            std::printf("MISMATCH: %s\n  want(%zu):", c.text.c_str(), c.ids.size());
            for (size_t k = 0; k < c.ids.size() && k < 24; ++k) std::printf(" %d", c.ids[k]);
            std::printf("\n  got (%zu):", got.size());
            for (size_t k = 0; k < got.size() && k < 24; ++k) std::printf(" %d", got[k]);
            std::printf("\n");
        }
    }
    std::printf("tokenizer fixtures: %zu cases, %d mismatches -> %s\n",
                cases.size(), failed, failed == 0 ? "PASS" : "FAIL");
    assert(failed == 0);
    return 0;
}
