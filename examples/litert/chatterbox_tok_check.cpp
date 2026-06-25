// Chatterbox tokenizer validation vs Python golden token IDs.
//
//   chatterbox_tok_check <bundle_dir> <golden_dir> <lang...>
//
// golden/{lang}.json carries "text", "lang", and "text_token_ids" = [255, ...BPE..., 0]
// (sot=255, eot=0 added by the wrapper). We compare encode(text,lang) to the middle.

#include "speech_core/models/chatterbox_tokenizer.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace speech_core;

static std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p);
    std::streamsize n = f.tellg(); f.seekg(0);
    std::string s((size_t)n, '\0'); f.read(&s[0], n); return s;
}
static std::vector<long> json_int_array(const std::string& s, const std::string& key) {
    size_t k = s.find("\"" + key + "\""); size_t lb = s.find('[', k), rb = s.find(']', lb);
    std::vector<long> out; std::string body = s.substr(lb + 1, rb - lb - 1), tok; size_t p = 0;
    while (p < body.size()) {
        size_t c = body.find(',', p); std::string t = body.substr(p, c == std::string::npos ? std::string::npos : c - p);
        size_t a = t.find_first_not_of(" \t\r\n");
        if (a != std::string::npos) out.push_back(std::stol(t.substr(a)));
        if (c == std::string::npos) break; p = c + 1;
    }
    return out;
}
static std::string json_str(const std::string& s, const std::string& key) {
    size_t k = s.find("\"" + key + "\""); size_t c = s.find(':', k); size_t q1 = s.find('"', c + 1), q2 = s.find('"', q1 + 1);
    return s.substr(q1 + 1, q2 - q1 - 1);
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <bundle_dir> <golden_dir> <lang...>\n", argv[0]); return 2; }
    std::string bdir = argv[1], gdir = argv[2];
    ChatterboxTokenizer tok(bdir + "/grapheme_mtl_merged_expanded_v1.json", bdir + "/Cangjie5_TC.json");
    std::printf("vocab=%zu\n", tok.vocab_size());
    int fails = 0;
    for (int a = 3; a < argc; ++a) {
        std::string lang = argv[a];
        std::string gj = read_file(gdir + "/" + lang + ".json");
        std::string text = json_str(gj, "text");
        auto gold = json_int_array(gj, "text_token_ids");   // [255, ...mid..., 0]
        std::vector<long> mid(gold.begin() + 1, gold.end() - 1);
        auto got = tok.encode(text, lang);
        bool ok = got.size() == mid.size();
        for (size_t i = 0; ok && i < mid.size(); ++i) ok = (got[i] == mid[i]);
        std::printf("[%s] text=\"%s\"\n  golden(mid)=%zu got=%zu  %s\n",
                    lang.c_str(), text.c_str(), mid.size(), got.size(), ok ? "MATCH" : "MISMATCH");
        if (!ok) {
            std::printf("  golden:"); for (long x : mid) std::printf(" %ld", x); std::printf("\n");
            std::printf("  got   :"); for (int x : got) std::printf(" %d", x); std::printf("\n");
            ++fails;
        }
    }
    std::printf(fails ? "TOK-CHECK-FAIL\n" : "TOK-CHECK-PASS\n");
    return fails ? 1 : 0;
}
