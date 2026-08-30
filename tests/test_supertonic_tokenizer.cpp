// SupertonicTokenizer chunking + window fitting. Pins the text front-end regressions from #140:
// greedy word-packing stranded the last word of a 57–62-codepoint sentence in its own chunk, the
// fragment was tokenized with a sentence-final "." and followed by 0.3 s of silence, and NFKD
// growth could push a chunk past the fixed text length so process() truncated it silently.
#include "speech_core/models/supertonic_tokenizer.h"

#include <cassert>
#include <cstdio>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace speech_core;

namespace {

// Identity table: id == codepoint across the BMP, so token ids read back as characters.
SupertonicTokenizer make_tokenizer() {
    std::vector<int32_t> table(0x10000);
    std::iota(table.begin(), table.end(), 0);
    return SupertonicTokenizer(std::move(table));
}

size_t codepoints(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s)
        if ((c & 0xC0) != 0x80) ++n;
    return n;
}

// Collapse whitespace runs to one space (chunk boundaries trim the newline/space between sentences).
std::string squash(const std::string& s) {
    std::string out;
    bool ws = false;
    for (char c : s) {
        const bool w = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
        if (w) { if (!ws && !out.empty()) out.push_back(' '); ws = true; }
        else   { out.push_back(c); ws = false; }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::string join(const std::vector<std::string>& v) {
    std::string out;
    for (const auto& s : v) { if (!out.empty()) out += ' '; out += s; }
    return out;
}

std::string join(const std::vector<SupertonicPiece>& v) {
    std::vector<std::string> t;
    for (const auto& p : v) t.push_back(p.text);
    return join(t);
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// The report's French paragraph, verbatim (line breaks as an LLM reply would carry them).
const char* kIssueText =
    "Bien sûr ! Voici une petite histoire de vacances d'été en 10 phrases :\n"
    "Cet été, j'ai passé mes vacances à la campagne. J'ai loué une petite maison près d'un lac "
    "magnifique. Chaque matin, je me levais tôt pour faire une longue promenade. J'ai beaucoup nagé "
    "dans l'eau fraîche et claire. J'ai aussi passé du temps à lire des livres sous les arbres. Un "
    "après-midi, j'ai aidé mes voisins à jardiner. Nous avons mangé des fruits frais cueillis dans "
    "le jardin. Le soir, je regardais les étoiles depuis ma terrasse. C'était un été très calme et "
    "reposant. Je suis revenu à la ville avec beaucoup de souvenirs heureux.\n"
    "Est-ce que tu veux que je traduise ce texte ou que je te raconte une autre histoire ?";

const std::vector<std::string> kIssueSentences = {
    "Bien sûr !",
    "Voici une petite histoire de vacances d'été en 10 phrases :",
    "Cet été, j'ai passé mes vacances à la campagne.",
    "J'ai loué une petite maison près d'un lac magnifique.",
    "Chaque matin, je me levais tôt pour faire une longue promenade.",
    "J'ai beaucoup nagé dans l'eau fraîche et claire.",
    "J'ai aussi passé du temps à lire des livres sous les arbres.",
    "Un après-midi, j'ai aidé mes voisins à jardiner.",
    "Nous avons mangé des fruits frais cueillis dans le jardin.",
    "Le soir, je regardais les étoiles depuis ma terrasse.",
    "C'était un été très calme et reposant.",
    "Je suis revenu à la ville avec beaucoup de souvenirs heureux.",
    "Est-ce que tu veux que je traduise ce texte ou que je te raconte une autre histoire ?",
};

// The packing budget LiteRTSupertonicTts derives for the published L=64 graph:
// int(64 * 3072 / 44100 s * 14 chars/s * 0.9) = 56 codepoints.
constexpr int kBudgetFr = 56;

// Stand-in for the duration predictor: one latent frame per codepoint (a frame is ~70 ms; the
// published 64-frame window holds ~4.5 s ≈ 60 codepoints of French).
int fake_frames(const std::string& t, bool /*continuation*/) {
    return static_cast<int>(codepoints(t));
}

// ---- chunk() ----

void test_issue_140_sentences_stay_whole() {
    auto tok = make_tokenizer();
    const auto chunks = tok.chunk(kIssueText, "fr", kBudgetFr);

    // Every sentence of the paragraph fits the text capacity, so none may be cut: each one appears
    // intact inside exactly one chunk (the old word-packer cut five of them at the budget).
    for (const auto& s : kIssueSentences) {
        int found = 0;
        for (const auto& c : chunks) if (contains(c, s)) ++found;
        if (found != 1) std::printf("    sentence not whole: %s\n", s.c_str());
        assert(found == 1);
    }
    // The stranded fragments the report lists must not exist as chunks.
    for (const char* orphan : {"arbres.", "jardin.", "heureux.", "promenade.", "campagne.",
                               "phrases :", "raconte une autre histoire ?"}) {
        for (const auto& c : chunks) assert(c != orphan);
    }
    // Every chunk still fits the model's text length after NFKD + <fr> wrap.
    for (const auto& c : chunks) assert(tok.wrapped_length(c, "fr") <= tok.max_text_tokens());
    // Nothing lost or reordered.
    assert(squash(join(chunks)) == squash(kIssueText));
    std::printf("  PASS: issue_140_sentences_stay_whole\n");
}

void test_short_sentences_still_pack() {
    // helper.py parity: sentences share a chunk while `len(cur) + 1 + len(s) <= max_len`.
    auto tok = make_tokenizer();
    const auto chunks = tok.chunk("Un. Deux. Trois quatre cinq.", "fr", 12);
    assert(chunks.size() == 2);
    assert(chunks[0] == "Un. Deux.");           // 3 + 1 + 5 = 9 ≤ 12
    assert(chunks[1] == "Trois quatre cinq.");  // 9 + 1 + 18 > 12
    // Exactly at the budget still packs (the old code charged a phantom leading space).
    const auto exact = tok.chunk("Un. Deux.", "fr", 9);
    assert(exact.size() == 1 && exact[0] == "Un. Deux.");
    std::printf("  PASS: short_sentences_still_pack\n");
}

void test_sentence_over_budget_kept_whole() {
    auto tok = make_tokenizer();
    const std::string s = "J'ai aussi passé du temps à lire des livres sous les arbres.";  // 60 > 56
    assert(codepoints(s) > kBudgetFr);
    const auto chunks = tok.chunk(s, "fr", kBudgetFr);
    assert(chunks.size() == 1);
    assert(chunks[0] == s);
    // ...and it does not glue onto a preceding short sentence either.
    const auto two = tok.chunk("Bien sûr ! " + s, "fr", kBudgetFr);
    assert(two.size() == 2 && two[0] == "Bien sûr !" && two[1] == s);
    std::printf("  PASS: sentence_over_budget_kept_whole\n");
}

void test_sentence_over_capacity_splits_balanced() {
    // Six 43-codepoint clauses = one 269-codepoint sentence, far past the 128-token text length.
    // It must be cut — but at commas, in balanced pieces, never leaving a stub.
    auto tok = make_tokenizer();
    const std::string clause = "the quick brown fox jumps over the lazy dog";
    std::string s;
    for (int i = 0; i < 6; ++i) { s += clause; s += (i == 5 ? "." : ", "); }
    const auto chunks = tok.chunk(s, "en", 0);
    assert(chunks.size() >= 3);
    for (const auto& c : chunks) {
        assert(tok.wrapped_length(c, "en") <= tok.max_text_tokens());
        assert(codepoints(c) >= 40);
        assert(c.back() == ',' || c.back() == '.');  // clause boundaries, not mid-phrase
    }
    assert(squash(join(chunks)) == s);
    std::printf("  PASS: sentence_over_capacity_splits_balanced\n");
}

void test_nfkd_growth_respects_text_capacity() {
    // 19 × "élève": 114 raw codepoints — under the old 118 raw-codepoint cap — but NFKD splits each
    // accent off (é → e + ◌́), so the wrapped form is 161 tokens and process() would have silently
    // dropped the end of the sentence.
    auto tok = make_tokenizer();
    std::string s;
    for (int i = 0; i < 19; ++i) { s += "élève"; s += (i == 18 ? "." : " "); }
    assert(codepoints(s) <= 118);
    assert(tok.wrapped_length(s, "fr") > tok.max_text_tokens());

    const auto chunks = tok.chunk(s, "fr", 0);
    assert(chunks.size() >= 2);
    for (const auto& c : chunks) {
        const int wrapped = tok.wrapped_length(c, "fr");
        assert(wrapped <= tok.max_text_tokens());
        const auto t = tok.process(c, "fr", tok.max_text_tokens());
        int real = 0;
        for (float m : t.mask) if (m > 0.0f) ++real;
        assert(real == wrapped);  // nothing truncated
    }
    assert(squash(join(chunks)) == s);
    std::printf("  PASS: nfkd_growth_respects_text_capacity\n");
}

void test_empty_and_unsupported_language() {
    auto tok = make_tokenizer();
    const auto empty = tok.chunk("", "fr", kBudgetFr);
    assert(empty.size() == 1 && empty[0].empty());
    bool threw = false;
    try { tok.chunk("Hallo.", "xx", kBudgetFr); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::printf("  PASS: empty_and_unsupported_language\n");
}

// ---- process() ----

void test_process_terminator_follows_continuation() {
    auto tok = make_tokenizer();
    // Wrapped form is "<fr>" + body + "</fr>": the body's last character sits 6 before the end.
    auto last_body_char = [&](const SupertonicTokenizer::Tokens& t) {
        int n = 0;
        for (float m : t.mask) if (m > 0.0f) ++n;
        return t.ids[static_cast<size_t>(n - 6)];
    };
    assert(last_body_char(tok.process("bonjour tout le monde", "fr", 128, false)) == '.');
    assert(last_body_char(tok.process("bonjour tout le monde", "fr", 128, true))  == ',');
    // Existing terminal punctuation is left alone either way.
    assert(last_body_char(tok.process("ça va ?", "fr", 128, true))   == '?');
    assert(last_body_char(tok.process("d'abord :", "fr", 128, true)) == ':');
    assert(last_body_char(tok.process("d'abord :", "fr", 128, false)) == ':');
    std::printf("  PASS: process_terminator_follows_continuation\n");
}

// ---- fit_to_window() ----

void test_fit_fits_untouched() {
    const std::string s = "Cet été, j'ai passé mes vacances à la campagne.";
    const auto pieces = SupertonicTokenizer::fit_to_window(s, fake_frames, 64);
    assert(pieces.size() == 1);
    assert(pieces[0].text == s);
    assert(!pieces[0].continuation);
    assert(!pieces[0].pause_before);
    std::printf("  PASS: fit_fits_untouched\n");
}

void test_fit_overflow_bisects_at_word_boundary() {
    // "... sous les" | "arbres." (#140): a 60-codepoint sentence over a 50-frame window must come
    // apart near its middle, on a word boundary, with the left half marked as continuing and no
    // pause before the right half.
    const std::string s = "J'ai aussi passé du temps à lire des livres sous les arbres.";
    const auto pieces = SupertonicTokenizer::fit_to_window(s, fake_frames, 50);
    assert(pieces.size() == 2);
    assert(codepoints(pieces[0].text) >= 20 && codepoints(pieces[1].text) >= 20);
    assert(pieces[1].text != "arbres.");
    assert(pieces[0].continuation);
    assert(!pieces[1].continuation);
    assert(!pieces[1].pause_before);
    // Words intact: the left piece is a prefix of the sentence that ends right before a space.
    assert(s.compare(0, pieces[0].text.size(), pieces[0].text) == 0);
    assert(s[pieces[0].text.size()] == ' ');
    assert(join(pieces) == s);
    std::printf("  PASS: fit_overflow_bisects_at_word_boundary\n");
}

void test_fit_prefers_clause_boundary() {
    const std::string s = "Chaque matin, je me levais tôt pour faire une longue promenade.";
    const auto pieces = SupertonicTokenizer::fit_to_window(s, fake_frames, 55);
    assert(pieces.size() == 2);
    assert(pieces[0].text == "Chaque matin,");
    assert(pieces[1].text == "je me levais tôt pour faire une longue promenade.");
    assert(pieces[0].continuation);   // "," is not a sentence end
    assert(!pieces[1].pause_before);
    std::printf("  PASS: fit_prefers_clause_boundary\n");
}

void test_fit_sentence_boundary_keeps_pause() {
    // Two packed sentences that overflow together split at ". ": the left keeps its own "." and
    // the right is flagged for the inter-sentence pause.
    const std::string a = "Cet été, j'ai passé mes vacances à la campagne.";
    const std::string b = "J'ai loué une petite maison près d'un lac.";
    const auto pieces = SupertonicTokenizer::fit_to_window(a + " " + b, fake_frames, 64);
    assert(pieces.size() == 2);
    assert(pieces[0].text == a);
    assert(pieces[1].text == b);
    assert(!pieces[0].continuation);
    assert(pieces[1].pause_before);
    std::printf("  PASS: fit_sentence_boundary_keeps_pause\n");
}

void test_fit_colon_newline_boundary() {
    // The report's first stranded fragment: "... en 10" | "phrases :". The lead-in and the first
    // story sentence form one 107-codepoint "sentence" (":" is not a sentence terminal) that must
    // come apart at the colon, not at a character count.
    const std::string lead  = "Voici une petite histoire de vacances d'été en 10 phrases :";
    const std::string first = "Cet été, j'ai passé mes vacances à la campagne.";
    const auto pieces = SupertonicTokenizer::fit_to_window(lead + "\n" + first, fake_frames, 64);
    assert(pieces.size() == 2);
    assert(pieces[0].text == lead);
    assert(pieces[1].text == first);
    std::printf("  PASS: fit_colon_newline_boundary\n");
}

void test_fit_measures_with_the_continuation_flag() {
    // The predictor must see the same continuation flag the synthesizer will tokenize with (the
    // trailing "," changes the ids), and every returned piece must have been measured as such.
    std::vector<std::pair<std::string, bool>> calls;
    auto recording = [&](const std::string& t, bool c) {
        calls.emplace_back(t, c);
        return static_cast<int>(codepoints(t));
    };
    const std::string s = "J'ai aussi passé du temps à lire des livres sous les arbres.";
    const auto pieces = SupertonicTokenizer::fit_to_window(s, recording, 50);
    assert(pieces.size() == 2);
    for (const auto& p : pieces) {
        bool seen = false;
        for (const auto& c : calls) if (c.first == p.text && c.second == p.continuation) seen = true;
        assert(seen);
    }
    assert(calls.front().first == s && !calls.front().second);
    std::printf("  PASS: fit_measures_with_the_continuation_flag\n");
}

void test_fit_too_short_returned_for_truncation() {
    // Under 2 × min_codepoints there is nothing to bisect: hand the piece back unchanged.
    const auto pieces = SupertonicTokenizer::fit_to_window("Bien sûr !", fake_frames, 4);
    assert(pieces.size() == 1);
    assert(pieces[0].text == "Bien sûr !");
    std::printf("  PASS: fit_too_short_returned_for_truncation\n");
}

void test_fit_depth_bound_terminates() {
    auto never_fits = [](const std::string&, bool) { return 1 << 20; };
    std::string s;
    for (int i = 0; i < 40; ++i) s += "word ";
    s += "end.";
    const auto pieces = SupertonicTokenizer::fit_to_window(s, never_fits, 64, 8, 3);
    assert(!pieces.empty() && pieces.size() <= 8);  // 2^3
    for (const auto& p : pieces) assert(codepoints(p.text) >= 8);
    assert(join(pieces) == s);
    std::printf("  PASS: fit_depth_bound_terminates\n");
}

}  // namespace

int main() {
    std::printf("test_supertonic_tokenizer\n");
    test_issue_140_sentences_stay_whole();
    test_short_sentences_still_pack();
    test_sentence_over_budget_kept_whole();
    test_sentence_over_capacity_splits_balanced();
    test_nfkd_growth_respects_text_capacity();
    test_empty_and_unsupported_language();
    test_process_terminator_follows_continuation();
    test_fit_fits_untouched();
    test_fit_overflow_bisects_at_word_boundary();
    test_fit_prefers_clause_boundary();
    test_fit_sentence_boundary_keeps_pause();
    test_fit_colon_newline_boundary();
    test_fit_measures_with_the_continuation_flag();
    test_fit_too_short_returned_for_truncation();
    test_fit_depth_bound_terminates();
    std::printf("ALL PASSED\n");
    return 0;
}
