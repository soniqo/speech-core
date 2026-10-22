// German text normalization and grapheme-to-phoneme conversion for the Kokoro
// German voices. See kokoro_german.h for the conventions. The structure is
// shared with the speech-swift port (GermanPhonemizer.swift): keep the two in
// lockstep when changing a rule.

#include "speech_core/models/kokoro_german.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace speech_core::german {

namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

using u32 = std::u32string;

u32 decode_utf8(const std::string& s) {
    u32 out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        char32_t cp = c;
        size_t len = 1;
        if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) { cp = c & 0x07; len = 4; }
        for (size_t k = 1; k < len; ++k) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string encode_utf8(const u32& s) {
    std::string out;
    for (char32_t cp : s) append_utf8(out, cp);
    return out;
}

char32_t first_codepoint(const std::string& s) {
    if (s.empty()) return 0;
    return decode_utf8(s.substr(0, 4))[0];
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(const u32& s, size_t pos, const u32& prefix) {
    if (pos + prefix.size() > s.size()) return false;
    return s.compare(pos, prefix.size(), prefix) == 0;
}

bool ends_with(const u32& s, const u32& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains(const u32& s, char32_t c) { return s.find(c) != u32::npos; }

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Stress marks as UTF-8.
const std::string PRIMARY = "ˈ";
const std::string SECONDARY = "ˌ";

std::string strip_stress(const std::string& ps) {
    return replace_all(replace_all(ps, PRIMARY, ""), SECONDARY, "");
}

std::string demote_primary(const std::string& ps) {
    return replace_all(ps, PRIMARY, SECONDARY);
}

std::string strip_marks_prefix(const std::string& ps) {
    // phonemes without leading stress marks
    std::string s = ps;
    while (true) {
        if (s.compare(0, PRIMARY.size(), PRIMARY) == 0) s.erase(0, PRIMARY.size());
        else if (s.compare(0, SECONDARY.size(), SECONDARY) == 0) s.erase(0, SECONDARY.size());
        else break;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Letters
// ---------------------------------------------------------------------------

bool is_vowel(char32_t c) {
    switch (c) {
        case U'a': case U'e': case U'i': case U'o': case U'u': case U'y':
        case U'ä': case U'ö': case U'ü':
            return true;
        default:
            return false;
    }
}

bool is_ascii_digit(char32_t c) { return c >= U'0' && c <= U'9'; }

bool is_letter(char32_t c) {
    if ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z')) return true;
    if (c == U'ß' || c == U'ẞ') return true;
    if (c >= 0xC0 && c <= 0x24F) return c != 0xD7 && c != 0xF7;  // Latin-1 / Extended-A/B letters
    return false;
}

bool is_upper(char32_t c) {
    return (c >= U'A' && c <= U'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7);
}

char32_t to_lower(char32_t c) {
    if (c >= U'A' && c <= U'Z') return c + 32;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;
    if (c == U'ẞ') return U'ß';
    if (c == U'İ') return U'i';  // Turkish dotted capital I (names)
    return c;
}

u32 lower(const u32& s) {
    u32 out = s;
    for (auto& c : out) c = to_lower(c);
    return out;
}

/// Fold loan-word accents onto the plain German letters the rules know.
u32 fold_accents(const u32& s) {
    u32 out;
    for (char32_t c : s) {
        switch (c) {
            case U'é': case U'è': case U'ê': case U'ë': out.push_back(U'e'); break;
            case U'á': case U'à': case U'â': case U'ã': out.push_back(U'a'); break;
            case U'ó': case U'ò': case U'ô': case U'õ': out.push_back(U'o'); break;
            case U'ú': case U'ù': case U'û': out.push_back(U'u'); break;
            case U'í': case U'ì': case U'î': case U'ï': out.push_back(U'i'); break;
            case U'ñ': out.push_back(U'n'); break;
            case U'ç': out.push_back(U's'); break;
            case U'ø': out.push_back(U'ö'); break;
            case U'å': out.push_back(U'o'); break;
            case U'æ': out.push_back(U'ä'); break;
            case U'œ': out.push_back(U'ö'); break;
            default: out.push_back(c);
        }
    }
    return out;
}

int count_nuclei(const u32& s) {
    int n = 0;
    size_t i = 0;
    while (i < s.size()) {
        if (is_vowel(s[i])) {
            ++n;
            while (i < s.size() && is_vowel(s[i])) ++i;
        } else {
            ++i;
        }
    }
    return n;
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

enum class PrefixKind { Unstressed, Stressed, Secondary };

struct Prefix {
    const char32_t* graph;
    const char* phon;
    PrefixKind kind;
};

// Order matters: longer prefixes first within a family (zusammen before zu).
const Prefix PREFIXES[] = {
    {U"zusammen", "ʦuːzˈamən", PrefixKind::Stressed}, {U"zurück", "ʦuːrˈʏk", PrefixKind::Stressed},
    {U"wieder", "viːdɜ", PrefixKind::Secondary}, {U"unter", "ˌʊntɜ", PrefixKind::Secondary},
    {U"über", "ˌyːbɜ", PrefixKind::Secondary},
    {U"durch", "dˈʊɐç", PrefixKind::Stressed}, {U"miss", "mˈɪs", PrefixKind::Stressed},
    {U"fort", "fˈɔɾt", PrefixKind::Stressed}, {U"nach", "nˈɑːx", PrefixKind::Stressed},
    {U"auf", "ˈWf", PrefixKind::Stressed}, {U"aus", "ˈWs", PrefixKind::Stressed},
    {U"bei", "bˈI", PrefixKind::Stressed}, {U"ein", "ˈIn", PrefixKind::Stressed},
    {U"mit", "mˈɪt", PrefixKind::Stressed}, {U"vor", "fˈoːɾ", PrefixKind::Stressed},
    {U"weg", "vˈɛk", PrefixKind::Stressed}, {U"her", "hˈɛɾ", PrefixKind::Stressed},
    {U"hin", "hˈɪn", PrefixKind::Stressed}, {U"los", "lˈoːs", PrefixKind::Stressed},
    {U"ab", "ˈap", PrefixKind::Stressed}, {U"an", "ˈan", PrefixKind::Stressed},
    {U"um", "ˈʊm", PrefixKind::Stressed}, {U"un", "ˈʊn", PrefixKind::Stressed},
    {U"ur", "ˈuːɾ", PrefixKind::Stressed}, {U"zu", "ʦˈuː", PrefixKind::Stressed},
    {U"ver", "fɛɾ", PrefixKind::Unstressed}, {U"zer", "ʦɛɾ", PrefixKind::Unstressed},
    {U"ent", "ɛnt", PrefixKind::Unstressed}, {U"emp", "ɛmp", PrefixKind::Unstressed},
    {U"be", "bə", PrefixKind::Unstressed}, {U"ge", "ɡə", PrefixKind::Unstressed},
    {U"er", "ɛɾ", PrefixKind::Unstressed},
};

struct Suffix {
    const char32_t* graph;
    const char* phon;
};

// Stressed loan suffixes; the phonemes carry the primary stress.
const Suffix STRESSED_SUFFIXES[] = {
    {U"ierungen", "ˈiːrʊŋən"}, {U"ierung", "ˈiːrʊŋ"}, {U"ierten", "ˈiːɾtən"}, {U"ierte", "ˈiːɾtə"},
    {U"ieren", "ˈiːrən"}, {U"ierst", "ˈiːɾst"}, {U"iert", "ˈiːɾt"}, {U"iere", "ˈiːrə"},
    {U"tionen", "ʦjˈoːnən"}, {U"tion", "ʦjˈoːn"}, {U"ssionen", "sjˈoːnən"}, {U"ssion", "sjˈoːn"},
    {U"sionen", "zjˈoːnən"}, {U"sion", "zjˈoːn"},
    {U"itäten", "iːtˈɛːtən"}, {U"ität", "iːtˈɛːt"}, {U"täten", "tˈɛːtən"}, {U"tät", "tˈɛːt"},
    {U"ismus", "ˈɪsmʊs"}, {U"istin", "ˈɪstɪn"}, {U"isten", "ˈɪstən"}, {U"ist", "ˈɪst"},
    {U"ieurs", "jˈøːɾs"}, {U"ieur", "jˈøːɾ"}, {U"euren", "ˈøːrən"}, {U"eure", "ˈøːrə"}, {U"eur", "ˈøːɾ"},
    {U"iven", "ˈiːvən"}, {U"ive", "ˈiːvə"}, {U"iv", "ˈiːf"},
    {U"agen", "ˈɑːʒən"}, {U"age", "ˈɑːʒə"},
    {U"ellen", "ˈɛlən"}, {U"elle", "ˈɛlə"}, {U"ell", "ˈɛl"},
    {U"enzen", "ˈɛnʦən"}, {U"enz", "ˈɛnʦ"}, {U"anzen", "ˈanʦən"}, {U"anz", "ˈanʦ"},
    {U"enten", "ˈɛntən"}, {U"ent", "ˈɛnt"}, {U"anten", "ˈantən"}, {U"ant", "ˈant"},
    {U"menten", "mˈɛntən"}, {U"ment", "mˈɛnt"},
    {U"uren", "ˈuːrən"}, {U"ur", "ˈuːɾ"},
    {U"inen", "ˈiːnən"}, {U"ine", "ˈiːnə"}, {U"ösen", "ˈøːzən"}, {U"öse", "ˈøːzə"}, {U"ös", "ˈøːs"},
    {U"eien", "ˈIən"}, {U"ei", "ˈI"}, {U"ien", "ˈiːən"}, {U"ie", "ˈiː"},
    {U"iken", "ˈiːkən"}, {U"ik", "ˈiːk"}, {U"ier", "ˈiːɾ"},
    {U"alen", "ˈɑːlən"}, {U"ale", "ˈɑːlə"}, {U"al", "ˈɑːl"}, {U"ären", "ˈɛːrən"}, {U"är", "ˈɛːɾ"},
    {U"ade", "ˈɑːdə"}, {U"ose", "ˈoːzə"}, {U"ette", "ˈɛtə"}, {U"ett", "ˈɛt"}, {U"een", "ˈeːən"}, {U"ee", "ˈeː"},
};
constexpr size_t MIN_STEM_BEFORE_SUFFIX = 3;

struct NativeSuffix {
    const char32_t* graph;
    const char* phon;
    bool may_take_secondary;
};

// Unstressed native suffixes with fixed pronunciations.
const NativeSuffix NATIVE_SUFFIXES[] = {
    // -ig-/-lich- always leave an unstressed syllable before -keit, so the
    // secondary stress is part of the suffix phonemes.
    {U"igkeiten", "ɪçkˌItən", false}, {U"igkeit", "ɪçkˌIt", false},
    {U"lichkeiten", "lɪçkˌItən", false}, {U"lichkeit", "lɪçkˌIt", false},
    {U"ungen", "ʊŋən", true}, {U"ung", "ʊŋ", true},
    {U"heiten", "hItən", true}, {U"heit", "hIt", true},
    {U"keiten", "kItən", true}, {U"keit", "kIt", true},
    {U"schaften", "ʃaftən", true}, {U"schaft", "ʃaft", true},
    {U"lichen", "lɪçən", false}, {U"liche", "lɪçə", false}, {U"lich", "lɪç", false},
    {U"ischen", "ɪʃən", false}, {U"ische", "ɪʃə", false}, {U"isch", "ɪʃ", false},
    {U"igen", "ɪɡən", false}, {U"ige", "ɪɡə", false}, {U"iger", "ɪɡɜ", false}, {U"ig", "ɪç", false},
    {U"chen", "çən", false}, {U"lein", "lIn", false},
    {U"baren", "bɑːrən", false}, {U"bare", "bɑːrə", false}, {U"bar", "bɑːɾ", false},
    {U"samen", "zɑːmən", false}, {U"same", "zɑːmə", false}, {U"sam", "zɑːm", false},
    {U"losen", "loːzən", false}, {U"lose", "loːzə", false}, {U"los", "loːs", false},
    {U"nisse", "nɪsə", false}, {U"nis", "nɪs", false}, {U"tums", "tuːms", false}, {U"tum", "tuːm", false},
    {U"lingen", "lɪŋən", false}, {U"lings", "lɪŋs", false}, {U"ling", "lɪŋ", false},
    {U"innen", "ɪnən", true}, {U"in", "ɪn", true},
};

struct Digraph {
    const char32_t* graph;
    const char* phon;
};
const Digraph DIGRAPHS[] = {
    {U"ei", "I"}, {U"ai", "I"}, {U"ey", "I"}, {U"ay", "I"}, {U"au", "W"},
    {U"eu", "ɔø"}, {U"äu", "ɔø"}, {U"ie", "iː"}, {U"aa", "ɑː"}, {U"ee", "eː"}, {U"oo", "oː"},
};

const char* digraph_phoneme(const u32& g) {
    for (const auto& d : DIGRAPHS) {
        if (g == d.graph) return d.phon;
    }
    return nullptr;
}

std::string long_vowel(char32_t c) {
    switch (c) {
        case U'a': return "ɑː"; case U'e': return "eː"; case U'i': return "iː";
        case U'o': return "oː"; case U'u': return "uː"; case U'ä': return "ɛː";
        case U'ö': return "øː"; case U'ü': return "yː"; case U'y': return "yː";
        default: return "";
    }
}

std::string short_vowel(char32_t c) {
    switch (c) {
        case U'a': return "a"; case U'e': return "ɛ"; case U'i': return "ɪ";
        case U'o': return "ɔ"; case U'u': return "ʊ"; case U'ä': return "ɛ";
        case U'ö': return "œ"; case U'ü': return "y"; case U'y': return "y";
        default: return "";
    }
}

// First code points of vowel phonemes, for context checks on neighbours.
bool phoneme_is_vowel(char32_t cp) {
    static const u32 vowels = U"aeiouyɑɛɪɔʊøœəɜIWA";
    return cp != 0 && contains(vowels, cp);
}

bool phoneme_is_back(char32_t cp) {
    static const u32 back = U"aɑɔʊuoW";
    return cp != 0 && contains(back, cp);
}

bool phoneme_is_front(char32_t cp) {
    static const u32 front = U"iɪeɛ";
    return cp != 0 && contains(front, cp);
}

// ---------------------------------------------------------------------------
// Prefix analysis
// ---------------------------------------------------------------------------

bool valid_stem(const u32& stem) {
    if (stem.size() < 3 || count_nuclei(stem) == 0) return false;
    return is_vowel(stem[0]) || stem.size() >= 4 || count_nuclei(stem) >= 2;
}

bool consonant_only_prefix(const u32& p) {
    return p == U"un" || p == U"ur" || p == U"ab" || p == U"ein" || p == U"an";
}

struct PrefixMatch {
    bool found = false;
    std::string phon;
    PrefixKind kind = PrefixKind::Unstressed;
    u32 stem;
};

PrefixMatch split_prefix(const u32& word) {
    PrefixMatch m;
    for (const auto& p : PREFIXES) {
        u32 graph(p.graph);
        if (!starts_with(word, 0, graph)) continue;
        u32 stem = word.substr(graph.size());
        if (!valid_stem(stem)) continue;
        if (consonant_only_prefix(graph) && is_vowel(stem[0])) {
            bool allowed = (graph == U"an" && stem[0] == U'e') ||
                           (graph == U"un" && stem[0] != U'i' && stem[0] != U'u');
            if (!allowed) continue;
        }
        m.found = true;
        m.phon = p.phon;
        m.kind = p.kind;
        m.stem = stem;
        if ((graph == U"her" || graph == U"hin") && is_vowel(stem[0])) {
            // herunter, hinein: the prefix is unstressed, the stem carries primary
            m.phon = strip_stress(m.phon);
            m.kind = PrefixKind::Unstressed;
        }
        return m;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Syllable nuclei
// ---------------------------------------------------------------------------

struct Nucleus {
    u32 grapheme;
    size_t start = 0;
    size_t end = 0;   // exclusive; includes a lengthening h
    bool long_h = false;
    std::string phoneme;
};

std::vector<Nucleus> find_nuclei(const u32& w) {
    std::vector<Nucleus> nuclei;
    size_t i = 0;
    const size_t n = w.size();
    while (i < n) {
        char32_t c = w[i];
        if (c == U'q' && i + 1 < n && w[i + 1] == U'u') { i += 2; continue; }
        bool vowel = is_vowel(c);
        if (c == U'y') {
            bool initial_glide = i == 0 && i + 1 < n && is_vowel(w[i + 1]);
            bool medial_glide = i > 0 && is_vowel(w[i - 1]) && i + 1 < n && is_vowel(w[i + 1]);
            if (initial_glide || medial_glide) vowel = false;
        }
        if (!vowel) { ++i; continue; }
        u32 g(1, c);
        if (i + 1 < n) {
            u32 two = w.substr(i, 2);
            if (digraph_phoneme(two) != nullptr) g = two;
        }
        size_t j = i + g.size();
        bool long_h = false;
        if (j < n && w[j] == U'h') { long_h = true; ++j; }
        Nucleus nu;
        nu.grapheme = g;
        nu.start = i;
        nu.end = j;
        nu.long_h = long_h;
        nuclei.push_back(nu);
        i = j;
    }
    return nuclei;
}

u32 consonants_after(const u32& w, const std::vector<Nucleus>& nuclei, size_t idx) {
    size_t end = idx + 1 < nuclei.size() ? nuclei[idx + 1].start : w.size();
    return w.substr(nuclei[idx].end, end - nuclei[idx].end);
}

int cluster_weight(const u32& cl) {
    if (cl.empty()) return 0;
    if (cl == U"th" || cl == U"ph") return 1;
    if (cl == U"ß") return 0;
    return static_cast<int>(cl.size());
}

std::string vowel_phoneme(const u32& w, const std::vector<Nucleus>& nuclei, size_t idx,
                          size_t stressed_idx, bool has_suffix) {
    const Nucleus& nu = nuclei[idx];
    const u32& g = nu.grapheme;
    const u32 cl = consonants_after(w, nuclei, idx);
    const bool last_in_core = idx + 1 == nuclei.size();
    const bool word_final_syllable = last_in_core && !has_suffix;
    const char32_t base = g[0];
    if (const char* d = digraph_phoneme(g)) return d;
    if (nu.long_h) return long_vowel(base);
    if (!cl.empty() && cl[0] == U'ß') return long_vowel(base);
    const bool post_stress = idx > stressed_idx;
    const bool pre_stress = idx < stressed_idx;
    const int weight = cluster_weight(cl);
    if (base == U'e' && post_stress) {
        // e + r before a consonant or at the word end -> vocalic r (Kinder,
        // Kindern, dauerte); e + r + vowel keeps a syllabic r (andere)
        if (!cl.empty() && cl[0] == U'r' && (cl.size() >= 2 || word_final_syllable)) return "ɜ";
        if (last_in_core && has_suffix) return "ə";
        if (weight >= 2 || word_final_syllable) return "ə";
        // open post-stress syllable: schwa before a schwa syllable ("redete",
        // "gegebenen"), otherwise tense e ("Kamera", "Lehrerinnen")
        if (idx + 1 < nuclei.size() && nuclei[idx + 1].grapheme == U"e" && cl.size() == 1 &&
            contains(U"tdnlrsmgbk", cl[0])) {
            return "ə";
        }
        return long_vowel(U'e');
    }
    if (base == U'i' && post_stress && weight >= 1) return "ɪ";
    if (base == U'u' && post_stress && word_final_syllable &&
        (cl == U"m" || cl == U"s" || cl == U"ms" || cl == U"ng" || cl == U"ngs" || cl == U"nft")) {
        return "ʊ";
    }
    // vowel + r + s(t): long (Kurs, erst, Vers)
    if (cl.size() >= 2 && cl[0] == U'r' && cl[1] == U's') return long_vowel(base);
    // u + r + consonant -> ʊɐ (Burg, kurz, durch); the r is absorbed
    if (base == U'u' && cl.size() >= 2 && cl[0] == U'r') return "ʊɐ";
    // u before ch: tense (Buch, Besuch, suchen) but not before chs/cht
    if (base == U'u' && starts_with(cl, 0, U"ch") && !starts_with(cl, 0, U"chs") &&
        !starts_with(cl, 0, U"cht")) {
        return "uː";
    }
    if (weight == 0) {
        if (has_suffix && last_in_core) {
            // "eskala|tion", "temporal|en": an open syllable before the suffix
            return base == U'a' ? "a" : long_vowel(base);
        }
        if (base == U'e') return idx > 0 ? "ə" : long_vowel(base);
        return long_vowel(base);
    }
    if (weight == 1) {
        if (pre_stress && base == U'a') return "a";
        return long_vowel(base);
    }
    return short_vowel(base);
}

// ---------------------------------------------------------------------------
// Consonants
// ---------------------------------------------------------------------------

u32 collapse_doubles(const u32& cl) {
    u32 out;
    for (char32_t ch : cl) {
        if (!out.empty() && out.back() == ch && contains(U"bdfgklmnprtz", ch)) continue;
        out.push_back(ch);
    }
    return out;
}

/// Map one consonant cluster. [prev_vowel]/[next_vowel] are the phonemes of
/// the surrounding nuclei ("" when none).
std::string map_consonants(const u32& cluster, const std::string& prev_vowel,
                           const std::string& next_vowel, bool at_word_start, bool at_stem_start) {
    // Keep "ss" distinct from a single s before collapsing doubles.
    u32 cl;
    for (size_t i = 0; i < cluster.size(); ++i) {
        if (cluster[i] == U's' && i + 1 < cluster.size() && cluster[i + 1] == U's') {
            cl.push_back(U'S');
            ++i;
        } else {
            cl.push_back(cluster[i]);
        }
    }
    cl = collapse_doubles(cl);
    const char32_t prev_cp = first_codepoint(prev_vowel);
    const char32_t next_cp = first_codepoint(next_vowel);
    const bool prev_back = phoneme_is_back(prev_cp) && prev_vowel != "ʊɐ";

    std::string out;
    size_t i = 0;
    const size_t n = cl.size();
    while (i < n) {
        const char32_t c = cl[i];
        const char32_t nxt = i + 1 < n ? cl[i + 1] : 0;
        const bool last = i + 1 >= n;
        if (starts_with(cl, i, U"tsch")) { out += (at_word_start && i == 0) ? "ʧ" : "tʃ"; i += 4; continue; }
        if (starts_with(cl, i, U"dsch")) { out += "ʤ"; i += 4; continue; }
        if (starts_with(cl, i, U"sch")) { out += "ʃ"; i += 3; continue; }
        if (starts_with(cl, i, U"sh")) { out += "ʃ"; i += 2; continue; }
        if (starts_with(cl, i, U"chs")) {
            out += (i == 0 && phoneme_is_back(prev_cp)) ? "xs" : "ks";
            i += 3;
            continue;
        }
        if (starts_with(cl, i, U"ch")) {
            if (i == 0 && at_word_start) {
                // Chemie, China -> ç before a front vowel; Chor, Christ -> k
                const bool ends_cluster = i + 2 >= n;
                out += (ends_cluster && phoneme_is_front(next_cp)) ? "ç" : "k";
            } else if (i == 0 && prev_back) {
                out += "x";
            } else {
                out += "ç";
            }
            i += 2;
            continue;
        }
        if (starts_with(cl, i, U"ck")) { out += "k"; i += 2; continue; }
        if (starts_with(cl, i, U"dt")) { out += "t"; i += 2; continue; }
        if (starts_with(cl, i, U"th")) { out += "t"; i += 2; continue; }
        if (starts_with(cl, i, U"ph")) { out += "f"; i += 2; continue; }
        if (starts_with(cl, i, U"pf")) { out += "pf"; i += 2; continue; }
        if (starts_with(cl, i, U"qu")) { out += "kv"; i += 2; continue; }
        if (starts_with(cl, i, U"tz")) { out += "ʦ"; i += 2; continue; }
        if (starts_with(cl, i, U"ng")) { out += "ŋ"; i += 2; continue; }
        if (starts_with(cl, i, U"nk")) { out += "ŋk"; i += 2; continue; }
        if (c == U'S') { out += "s"; ++i; continue; }
        if (starts_with(cl, i, U"sp") && i == 0 && at_stem_start) { out += "ʃp"; i += 2; continue; }
        if (starts_with(cl, i, U"st") && i == 0 && at_stem_start) { out += "ʃt"; i += 2; continue; }

        const bool onset_before_vowel = last && !next_vowel.empty();
        const bool voiced_ok = (!last && nxt != 0 && contains(U"lrnmvw", nxt)) || onset_before_vowel;
        const char32_t before = out.empty() ? 0 : decode_utf8(out).back();
        switch (c) {
            case U'b': out += voiced_ok ? "b" : "p"; break;
            case U'd': out += voiced_ok ? "d" : "t"; break;
            case U'g': out += voiced_ok ? "ɡ" : "k"; break;
            case U's': {
                static const u32 voiceless = U"kptfçʃx";
                out += (onset_before_vowel && !(before != 0 && contains(voiceless, before))) ? "z" : "s";
                break;
            }
            case U'ß': out += "s"; break;
            case U'v':
                out += ((i == 0 && (at_word_start || at_stem_start)) || !onset_before_vowel) ? "f" : "v";
                break;
            case U'w': out += "v"; break;
            case U'z': out += "ʦ"; break;
            case U'c': {
                static const u32 front_letters = U"eiyäöü";
                static const u32 front_phonemes = U"eɛiɪyø";
                bool front = (nxt != 0 && contains(front_letters, nxt)) ||
                             (last && next_cp != 0 && contains(front_phonemes, next_cp));
                out += front ? "ʦ" : "k";
                break;
            }
            case U'x': out += "ks"; break;
            case U'y': out += "j"; break;
            case U'r':
                if (onset_before_vowel) {
                    out += (i == 0 || cl[i - 1] == U'f' || ends_with(out, std::string("ʃ"))) ? "r" : "ɾ";
                } else {
                    out += "ɾ";
                }
                break;
            default:
                append_utf8(out, c);
        }
        ++i;
    }
    return out;
}

/// Insert a stress mark before the first vowel phoneme (espeak puts the
/// mark before the vowel, after any onset consonants: "ʃˌaft").
std::string mark_before_first_vowel(const std::string& phon, const std::string& mark) {
    u32 cps = decode_utf8(phon);
    std::string out;
    bool placed = false;
    for (char32_t cp : cps) {
        if (!placed && phoneme_is_vowel(cp)) {
            out += mark;
            placed = true;
        }
        append_utf8(out, cp);
    }
    return placed ? out : mark + phon;
}

std::string suffix_first_vowel(const std::string& phon) {
    std::string bare = strip_marks_prefix(phon);
    char32_t cp = first_codepoint(bare);
    if (!phoneme_is_vowel(cp)) return "";
    std::string s;
    append_utf8(s, cp);
    return s;
}

/// Phonemize a core (prefix and suffix stripped).
std::string stem_g2p(const u32& core, size_t stressed_idx, const std::string& suffix_phon,
                     bool at_word_start, bool at_stem_start, bool secondary_on_first) {
    std::vector<Nucleus> nuclei = find_nuclei(core);
    const bool has_suffix = !suffix_phon.empty();
    const std::string suffix_first = has_suffix ? suffix_first_vowel(suffix_phon) : "";
    if (nuclei.empty()) {
        return map_consonants(core, "", suffix_first, at_word_start, at_stem_start);
    }
    for (size_t idx = 0; idx < nuclei.size(); ++idx) {
        nuclei[idx].phoneme = vowel_phoneme(core, nuclei, idx, stressed_idx, has_suffix);
    }
    std::string out = map_consonants(core.substr(0, nuclei[0].start), "", nuclei[0].phoneme,
                                     at_word_start, at_stem_start);
    for (size_t idx = 0; idx < nuclei.size(); ++idx) {
        const Nucleus& nu = nuclei[idx];
        if (idx == stressed_idx) out += PRIMARY;
        else if (idx == 0 && secondary_on_first) out += SECONDARY;
        out += nu.phoneme;
        u32 cl = consonants_after(core, nuclei, idx);
        if ((nu.phoneme == "ʊɐ" || nu.phoneme == "ɜ") && !cl.empty() && cl[0] == U'r') cl.erase(0, 1);
        const std::string nxt = idx + 1 < nuclei.size() ? nuclei[idx + 1].phoneme : suffix_first;
        out += map_consonants(cl, nu.phoneme, nxt, false, false);
    }
    return out;
}

std::string g2p_lower(const u32& w, int depth) {
    if (w.empty()) return "";
    PrefixMatch pm = split_prefix(w);
    if (pm.found && depth < 3) {
        // The stem is phonemized as a word of its own (so sp/st, v, initial r
        // and ch follow the word-initial rules) and may carry further
        // prefixes: aus-ge-wählt, be-vor-stehen, un-er-schütterlich.
        std::string body = g2p_lower(pm.stem, depth + 1);
        if (pm.kind == PrefixKind::Stressed) return pm.phon + demote_primary(body);
        return pm.phon + body;
    }
    const u32& stem = w;
    // Stressed loan suffix
    std::string suffix_phon;
    u32 suffix_graph;
    for (const auto& s : STRESSED_SUFFIXES) {
        u32 graph(s.graph);
        if (ends_with(stem, graph) && stem.size() - graph.size() >= MIN_STEM_BEFORE_SUFFIX &&
            count_nuclei(stem.substr(0, stem.size() - graph.size())) >= 1) {
            suffix_graph = graph;
            suffix_phon = s.phon;
            break;
        }
    }
    // Unstressed native suffix
    std::string native_phon;
    u32 native_graph;
    bool native_secondary = false;
    if (suffix_graph.empty()) {
        for (const auto& s : NATIVE_SUFFIXES) {
            u32 graph(s.graph);
            if (!ends_with(stem, graph)) continue;
            u32 core_candidate = stem.substr(0, stem.size() - graph.size());
            if (core_candidate.size() < 2 || count_nuclei(core_candidate) < 1 ||
                is_vowel(core_candidate.back())) {
                continue;
            }
            if (graph == U"chen" && core_candidate.back() == U's') {
                // waschen, löschen: that is "sch" + "en"; Häuschen keeps the
                // diminutive thanks to its umlaut
                bool umlaut = contains(core_candidate, U'ä') || contains(core_candidate, U'ö') ||
                              contains(core_candidate, U'ü');
                if (!umlaut) continue;
            }
            native_graph = graph;
            native_phon = s.phon;
            native_secondary = s.may_take_secondary;
            break;
        }
    }
    u32 core = stem;
    if (!suffix_graph.empty()) core = stem.substr(0, stem.size() - suffix_graph.size());
    else if (!native_graph.empty()) core = stem.substr(0, stem.size() - native_graph.size());

    const size_t nuclei_count = find_nuclei(core).size();
    // The suffix carries the primary stress when it is a loan suffix.
    const size_t stressed_idx = suffix_graph.empty() ? 0 : nuclei_count;
    const bool secondary_on_first = !suffix_graph.empty() && nuclei_count >= 2;
    const std::string& suffix_for_core = !suffix_phon.empty() ? suffix_phon : native_phon;
    std::string body = stem_g2p(core, stressed_idx, suffix_for_core, true, true, secondary_on_first);
    if (!suffix_phon.empty()) {
        body += suffix_phon;
    } else if (!native_phon.empty()) {
        // -ung/-heit/-keit/-schaft/-in carry secondary stress when at least
        // one unstressed syllable separates them from the primary
        // (Wissenschaft, Lehrerin, Beobachtung) but not when the primary sits
        // right before them (Beratung, Landschaft, Gräfin).
        if (native_secondary && nuclei_count >= 1 && nuclei_count - 1 - stressed_idx >= 1) {
            body += mark_before_first_vowel(native_phon, SECONDARY);
        } else {
            body += native_phon;
        }
    }
    return body;
}

// ---------------------------------------------------------------------------
// Number normalization
// ---------------------------------------------------------------------------

const char* ONES[] = {"", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
                      "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn",
                      "siebzehn", "achtzehn", "neunzehn"};
const char* TENS[] = {"", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig",
                      "achtzig", "neunzig"};
const char* MONTHS[] = {"", "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
                        "September", "Oktober", "November", "Dezember"};

std::string int_to_de(long long n, bool standalone = true) {
    if (n < 0) return "minus " + int_to_de(-n, standalone);
    if (n == 0) return "null";
    if (n == 1) return standalone ? "eins" : "ein";
    if (n < 20) return ONES[n];
    if (n < 100) {
        long long ones = n % 10, tens = n / 10;
        if (ones) return std::string(ONES[ones]) + "und" + TENS[tens];
        return TENS[tens];
    }
    if (n < 1000) {
        long long h = n / 100, r = n % 100;
        return std::string(ONES[h]) + "hundert" + (r ? int_to_de(r, false) : "");
    }
    if (n < 1000000) {
        long long t = n / 1000, r = n % 1000;
        std::string prefix = t != 1 ? int_to_de(t, false) : "ein";
        return prefix + "tausend" + (r ? int_to_de(r, false) : "");
    }
    struct Scale { long long divisor; const char* singular; const char* plural; };
    static const Scale scales[] = {
        {1000000000000LL, "eine Billion", "Billionen"},
        {1000000000LL, "eine Milliarde", "Milliarden"},
        {1000000LL, "eine Million", "Millionen"},
    };
    for (const auto& s : scales) {
        if (n >= s.divisor) {
            long long count = n / s.divisor, r = n % s.divisor;
            std::string word = count == 1 ? s.singular : int_to_de(count, false) + " " + s.plural;
            return word + (r ? " " + int_to_de(r, false) : "");
        }
    }
    return std::to_string(n);
}

std::string ordinal_stem(long long n) {
    switch (n) {
        case 1: return "erst"; case 2: return "zweit"; case 3: return "dritt";
        case 7: return "siebt"; case 8: return "acht";
        default: break;
    }
    std::string stem = int_to_de(n, false) + (n < 20 ? "t" : "st");
    if (n == 100 || n == 1000) stem = stem.substr(3);  // drop leading "ein"
    return stem;
}

std::string year_to_de(long long n) {
    if (n >= 1100 && n <= 1999) {
        long long c = n / 100, r = n % 100;
        return int_to_de(c, false) + "hundert" + (r ? int_to_de(r, false) : "");
    }
    return int_to_de(n);
}

std::string digits_to_de(const u32& digits) {
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i) out += " ";
        out += int_to_de(digits[i] - U'0');
    }
    return out;
}

long long to_number(const u32& digits) {
    long long v = 0;
    for (char32_t d : digits) {
        if (v > 100000000000000LL) break;  // keep absurd lengths finite
        v = v * 10 + (d - U'0');
    }
    return v;
}

const char* currency_name(char32_t sym) {
    switch (sym) {
        case U'€': return "Euro"; case U'$': return "Dollar";
        case U'£': return "Pfund"; case U'¥': return "Yen";
        default: return nullptr;
    }
}

/// [whole] digits and [frac] digits (possibly empty) of an amount.
std::string currency_words(char32_t sym, const u32& whole, const u32& frac) {
    long long euros = whole.empty() ? 0 : to_number(whole);
    long long cents = 0;
    if (!frac.empty()) {
        u32 two = frac.substr(0, 2);
        while (two.size() < 2) two.push_back(U'0');
        cents = to_number(two);
    }
    std::string word = currency_name(sym);
    if (cents == 0) return int_to_de(euros, false) + " " + word;
    return int_to_de(euros, false) + " " + word + " und " + int_to_de(cents, false) + " Cent";
}

struct Unit { const char32_t* symbol; const char* spoken; };
// Longest first so "kg" wins over "g".
const Unit UNITS[] = {
    {U"kWh", "Kilowattstunden"}, {U"GHz", "Gigahertz"}, {U"MHz", "Megahertz"}, {U"kHz", "Kilohertz"},
    {U"km/h", "Kilometer pro Stunde"}, {U"°C", "Grad Celsius"}, {U"mAh", "Milliamperestunden"},
    {U"mg", "Milligramm"}, {U"kg", "Kilogramm"}, {U"km", "Kilometer"}, {U"cm", "Zentimeter"},
    {U"mm", "Millimeter"}, {U"ml", "Milliliter"}, {U"Hz", "Hertz"}, {U"g", "Gramm"}, {U"l", "Liter"},
    {U"m", "Meter"}, {U"°", "Grad"},
};

bool is_alnum(char32_t c) { return is_letter(c) || is_ascii_digit(c); }

/// Unit symbol starting at [i] that ends the word: (length, spoken form).
std::pair<size_t, const char*> unit_at(const u32& text, size_t i) {
    for (const auto& u : UNITS) {
        u32 sym(u.symbol);
        if (starts_with(text, i, sym)) {
            size_t end = i + sym.size();
            if (end >= text.size() || !is_alnum(text[end])) return {sym.size(), u.spoken};
        }
    }
    return {0, nullptr};
}

u32 read_digits(const u32& text, size_t i, size_t& end) {
    size_t j = i;
    while (j < text.size() && is_ascii_digit(text[j])) ++j;
    end = j;
    return text.substr(i, j - i);
}

/// Lower-cased alphabetic word ending right before [i] (one space allowed).
u32 word_before(const u32& text, size_t i) {
    size_t j = i;
    if (j > 0 && text[j - 1] == U' ') --j;
    size_t k = j;
    while (k > 0 && is_letter(text[k - 1])) --k;
    return lower(text.substr(k, j - k));
}

struct Abbreviation { const char32_t* abbr; const char* full; bool needs_digit; };
const Abbreviation ABBREVIATIONS[] = {
    {U"Dr.", "Doktor", false}, {U"Prof.", "Professor", false}, {U"Hr.", "Herr", false},
    {U"Fr.", "Frau", false}, {U"Str.", "Straße", false}, {U"Nr.", "Nummer", true},
    {U"Tel.", "Telefon", false}, {U"Abt.", "Abteilung", false}, {U"Abs.", "Absatz", true},
    {U"z.B.", "zum Beispiel", false}, {U"z. B.", "zum Beispiel", false},
    {U"d.h.", "das heißt", false}, {U"d. h.", "das heißt", false},
    {U"u.a.", "unter anderem", false}, {U"u. a.", "unter anderem", false},
    {U"bzw.", "beziehungsweise", false}, {U"usw.", "und so weiter", false},
    {U"etc.", "et cetera", false}, {U"ca.", "circa", false}, {U"vgl.", "vergleiche", false},
    {U"inkl.", "inklusive", false}, {U"exkl.", "exklusive", false}, {U"ggf.", "gegebenenfalls", false},
    {U"i.d.R.", "in der Regel", false}, {U"o.ä.", "oder ähnliches", false}, {U"u.U.", "unter Umständen", false},
    {U"Jan.", "Januar", false}, {U"Feb.", "Februar", false}, {U"Mär.", "März", false}, {U"Apr.", "April", false},
    {U"Jun.", "Juni", false}, {U"Jul.", "Juli", false}, {U"Aug.", "August", false}, {U"Sep.", "September", false},
    {U"Okt.", "Oktober", false}, {U"Nov.", "November", false}, {U"Dez.", "Dezember", false},
    {U"GmbH", "Gesellschaft mit beschränkter Haftung", false}, {U"AG", "Aktiengesellschaft", false},
};

bool at_word_boundary(const u32& text, size_t i) {
    return i == 0 || !is_alnum(text[i - 1]);
}

u32 expand_abbreviations(u32 text) {
    for (const auto& a : ABBREVIATIONS) {
        u32 abbr(a.abbr);
        u32 full = decode_utf8(a.full);
        size_t start = 0;
        while (true) {
            size_t pos = text.find(abbr, start);
            if (pos == u32::npos) break;
            size_t end = pos + abbr.size();
            bool ok = at_word_boundary(text, pos);
            if (abbr.back() == U'.') {
                ok = ok && (end >= text.size() || text[end] == U' ' || text[end] == U'\n');
            } else {
                ok = ok && (end >= text.size() || !is_alnum(text[end]));
            }
            if (a.needs_digit) {
                size_t k = end;
                while (k < text.size() && text[k] == U' ') ++k;
                ok = ok && k < text.size() && is_ascii_digit(text[k]);
            }
            if (ok) {
                text = text.substr(0, pos) + full + text.substr(end);
                start = pos + full.size();
            } else {
                start = end;
            }
        }
    }
    return text;
}

u32 normalize_numbers(const u32& text) {
    u32 out;
    const size_t n = text.size();
    size_t i = 0;
    auto emit = [&](const std::string& s) { out += decode_utf8(s); };
    while (i < n) {
        const char32_t c = text[i];
        // currency symbol before the amount: € 12,50
        if (currency_name(c) != nullptr && i + 1 < n) {
            size_t j = i + 1;
            while (j < n && text[j] == U' ') ++j;
            size_t k;
            u32 digits = read_digits(text, j, k);
            if (!digits.empty()) {
                u32 whole = digits, frac;
                while (k < n && (text[k] == U'.' || text[k] == U',') && k + 1 < n && is_ascii_digit(text[k + 1])) {
                    size_t k2;
                    u32 more = read_digits(text, k + 1, k2);
                    if (text[k] == U',') frac = more; else whole += more;
                    k = k2;
                }
                emit(currency_words(c, whole, frac));
                i = k;
                continue;
            }
        }
        if (!is_ascii_digit(c) || (i > 0 && is_letter(text[i - 1]))) {
            out.push_back(c);
            ++i;
            continue;
        }
        size_t j;
        u32 digits = read_digits(text, i, j);
        // time HH:MM
        if (j < n && text[j] == U':' && digits.size() <= 2) {
            size_t k;
            u32 mins = read_digits(text, j + 1, k);
            if (mins.size() == 2 && (k >= n || !is_ascii_digit(text[k]))) {
                long long h = to_number(digits), m = to_number(mins);
                if (h <= 23 && m <= 59) {
                    if (starts_with(text, k, U" Uhr")) k += 4;
                    emit(int_to_de(h) + " Uhr" + (m ? " " + int_to_de(m) : ""));
                    i = k;
                    continue;
                }
            }
        }
        // date DD.MM.YYYY
        if (j + 1 < n && text[j] == U'.' && digits.size() <= 2 && is_ascii_digit(text[j + 1])) {
            size_t k;
            u32 month = read_digits(text, j + 1, k);
            if (month.size() <= 2 && k + 1 < n && text[k] == U'.' && is_ascii_digit(text[k + 1])) {
                size_t k2;
                u32 year = read_digits(text, k + 1, k2);
                if (year.size() == 4 && (k2 >= n || !is_ascii_digit(text[k2]))) {
                    long long d = to_number(digits), mo = to_number(month), y = to_number(year);
                    if (d >= 1 && d <= 31 && mo >= 1 && mo <= 12) {
                        u32 prev = word_before(out, out.size());
                        std::string suffix = "er";
                        if (prev == U"am" || prev == U"im" || prev == U"vom" || prev == U"zum" || prev == U"den") suffix = "en";
                        else if (prev == U"der") suffix = "e";
                        emit(ordinal_stem(d) + suffix + " " + MONTHS[mo] + " " + year_to_de(y));
                        i = k2;
                        continue;
                    }
                }
            }
        }
        // grouped thousands 1.234.567(,89)
        auto group_at = [&](size_t k) {
            return k + 3 < n && text[k] == U'.' && is_ascii_digit(text[k + 1]) && is_ascii_digit(text[k + 2]) &&
                   is_ascii_digit(text[k + 3]) && (k + 4 >= n || !is_ascii_digit(text[k + 4]));
        };
        if (digits.size() <= 3 && group_at(j)) {
            u32 whole = digits;
            size_t k = j;
            while (group_at(k)) {
                whole += text.substr(k + 1, 3);
                k += 4;
            }
            u32 frac;
            if (k + 1 < n && text[k] == U',' && is_ascii_digit(text[k + 1])) frac = read_digits(text, k + 1, k);
            emit(int_to_de(to_number(whole)) + (frac.empty() ? "" : " Komma " + digits_to_de(frac)));
            i = k;
            continue;
        }
        // decimal comma 3,14 (optionally followed by a unit or currency)
        if (j + 1 < n && text[j] == U',' && is_ascii_digit(text[j + 1])) {
            size_t k;
            u32 frac = read_digits(text, j + 1, k);
            size_t k2 = k;
            while (k2 < n && text[k2] == U' ') ++k2;
            if (k2 < n && currency_name(text[k2]) != nullptr) {
                emit(currency_words(text[k2], digits, frac));
                i = k2 + 1;
                continue;
            }
            emit(int_to_de(to_number(digits)) + " Komma " + digits_to_de(frac));
            auto unit = unit_at(text, k2);
            if (unit.first) {
                emit(std::string(" ") + unit.second);
                i = k2 + unit.first;
            } else {
                i = k;
            }
            continue;
        }
        // currency or unit after the amount: 12 €, 500 g, 1,5 kg
        {
            size_t k = j;
            while (k < n && text[k] == U' ') ++k;
            if (k < n && currency_name(text[k]) != nullptr) {
                emit(currency_words(text[k], digits, u32()));
                i = k + 1;
                continue;
            }
            auto unit = unit_at(text, k);
            if (unit.first) {
                emit(int_to_de(to_number(digits)) + " " + unit.second);
                i = k + unit.first;
                continue;
            }
        }
        // ordinal "3." before a word
        if (j + 2 < n && text[j] == U'.' && text[j + 1] == U' ' && is_letter(text[j + 2])) {
            u32 prev = word_before(out, out.size());
            emit(ordinal_stem(to_number(digits)) + (prev == U"am" ? "en" : "e"));
            i = j + 1;
            continue;
        }
        // phone-like digit groups: 0170 123456
        if (digits.size() >= 2 && digits.size() <= 4 && j + 1 < n && (text[j] == U' ' || text[j] == U'-') &&
            is_ascii_digit(text[j + 1])) {
            std::vector<u32> groups{digits};
            size_t k = j;
            bool ok = true;
            while (k + 1 < n && (text[k] == U' ' || text[k] == U'-') && is_ascii_digit(text[k + 1])) {
                size_t k2;
                u32 more = read_digits(text, k + 1, k2);
                if (more.size() < 2 || more.size() > 6) { ok = false; break; }
                groups.push_back(more);
                k = k2;
            }
            if (ok && groups.size() >= 2) {
                std::string spoken;
                for (size_t g = 0; g < groups.size(); ++g) {
                    if (g) spoken += " ";
                    spoken += digits_to_de(groups[g]);
                }
                emit(spoken);
                i = k;
                continue;
            }
        }
        // year
        if (digits.size() == 4) {
            long long y = to_number(digits);
            if (y >= 1100 && y <= 2099) {
                emit(year_to_de(y));
                i = j;
                continue;
            }
        }
        emit(int_to_de(to_number(digits)));
        i = j;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Word-level fallbacks over the dictionary
// ---------------------------------------------------------------------------

struct Inflection { const char32_t* graph; const char* phon; bool vowel_initial; };
// Tried shortest suffix first (longest stem wins).
const Inflection INFLECTIONS[] = {
    {U"e", "ə", true}, {U"n", "n", false}, {U"s", "s", false}, {U"t", "t", false},
    {U"er", "ɜ", true}, {U"te", "tə", false}, {U"st", "st", false}, {U"en", "ən", true},
    {U"em", "əm", true}, {U"es", "əs", true}, {U"et", "ət", true},
    {U"ung", "ʊŋ", true}, {U"ern", "ɜn", true}, {U"ers", "ɜs", true}, {U"ert", "ɜt", true},
    {U"est", "əst", true}, {U"ten", "tən", false}, {U"tes", "təs", false}, {U"tet", "tət", false},
    {U"ungen", "ʊŋən", true}, {U"innen", "ɪnən", true},
};

const char32_t* LINKING[] = {U"", U"s", U"n", U"en", U"es", U"er", U"e"};
const char* LINK_PHONEMES[] = {"", "s", "n", "ən", "əs", "ɜ", "ə"};

const std::string* dict_find(const Dictionary& dict, const u32& key) {
    auto it = dict.find(encode_utf8(key));
    return it == dict.end() ? nullptr : &it->second;
}

/// A final devoiced consonant becomes voiced again before a vowel suffix when
/// the spelling ends in b/d/g/s (Tag -> Tage, Haus -> Häuser).
std::string revoice(const std::string& stem_phon, const u32& stem_graph) {
    if (stem_phon.empty() || stem_graph.empty()) return stem_phon;
    char32_t last_letter = stem_graph.back();
    if (last_letter != U'b' && last_letter != U'd' && last_letter != U'g' && last_letter != U's') return stem_phon;
    if (ends_with(stem_graph, U"ss") || last_letter == U'ß') return stem_phon;
    char last = stem_phon.back();
    switch (last) {
        case 'p': return stem_phon.substr(0, stem_phon.size() - 1) + "b";
        case 't': return stem_phon.substr(0, stem_phon.size() - 1) + "d";
        case 'k': return stem_phon.substr(0, stem_phon.size() - 1) + "ɡ";
        case 's': return stem_phon.substr(0, stem_phon.size() - 1) + "z";
        default: return stem_phon;
    }
}

std::string join_suffix(const std::string& stem_phon, const u32& stem_graph, const std::string& phon,
                        bool vowel_initial) {
    std::string sp = stem_phon;
    if (vowel_initial) {
        sp = revoice(sp, stem_graph);
        if (ends_with(sp, std::string("ɜ"))) sp = sp.substr(0, sp.size() - 2) + "ər";  // geringer -> geringere
        if (ends_with(stem_graph, U"ig") && ends_with(sp, std::string("ɪç"))) {
            sp = sp.substr(0, sp.size() - 2) + "ɡ";  // willig -> willigen
        }
    } else {
        if (!sp.empty()) {
            if (sp.back() == 'b') sp.back() = 'p';
            else if (sp.back() == 'd') sp.back() = 't';
            else if (sp.back() == 'z') sp.back() = 's';
            else if (ends_with(sp, std::string("ɡ"))) sp = sp.substr(0, sp.size() - 2) + "k";
        }
        if (!sp.empty() && sp.back() == 'r') sp = sp.substr(0, sp.size() - 1) + "ɾ";  // kritisieren -> kritisierst
    }
    return sp + phon;
}

bool inflected(const u32& word, const Dictionary& dict, std::string& out) {
    for (const auto& inf : INFLECTIONS) {
        u32 graph(inf.graph);
        if (!ends_with(word, graph) || word.size() - graph.size() < 3) continue;
        u32 stem = word.substr(0, word.size() - graph.size());
        if (count_nuclei(stem) == 0) continue;
        if (const std::string* base = dict_find(dict, stem)) {
            out = join_suffix(*base, stem, inf.phon, inf.vowel_initial);
            return true;
        }
        if (const std::string* infinitive = dict_find(dict, stem + U"en")) {
            if (ends_with(*infinitive, std::string("ən"))) {
                out = join_suffix(infinitive->substr(0, infinitive->size() - 3), stem, inf.phon,
                                  inf.vowel_initial);
                return true;
            }
        }
    }
    return false;
}

bool compound(const u32& word, const Dictionary& dict, std::string& out, int depth) {
    if (depth > 2 || word.size() < 7) return false;
    for (size_t cut = word.size() - 3; cut > 3; --cut) {
        u32 head = word.substr(0, cut);
        const std::string* hp = dict_find(dict, head);
        if (hp == nullptr || count_nuclei(head) == 0) continue;
        for (size_t li = 0; li < sizeof(LINKING) / sizeof(LINKING[0]); ++li) {
            u32 link(LINKING[li]);
            u32 rest = word.substr(cut);
            if (!link.empty() && !starts_with(rest, 0, link)) continue;
            u32 tail = rest.substr(link.size());
            if (tail.size() < 3 || count_nuclei(tail) == 0) continue;
            std::string tp;
            if (const std::string* direct = dict_find(dict, tail)) tp = *direct;
            else if (inflected(tail, dict, tp)) {}
            else if (tail.size() >= 7 && compound(tail, dict, tp, depth + 1)) {}
            else continue;
            // The head keeps its primary stress; the tail's primary becomes
            // secondary when the compound has three or more syllables
            // (Eisenbahn, Hauptbahnhof) and is dropped otherwise (Bahnhof).
            std::string tail_phon = count_nuclei(word) >= 3 ? demote_primary(tp) : strip_stress(tp);
            std::string head_phon = *hp;
            const bool vowel_follows = link.empty()
                ? phoneme_is_vowel(first_codepoint(strip_marks_prefix(tail_phon)))
                : is_vowel(link[0]);
            if (vowel_follows) {
                head_phon = revoice(head_phon, head);  // Kind|es|haus, Tag|e|buch
                if (ends_with(head_phon, std::string("ɜ"))) {
                    head_phon = head_phon.substr(0, head_phon.size() - 2) + "ər";  // Kleider|ordnung
                }
            }
            out = head_phon + LINK_PHONEMES[li] + tail_phon;
            return true;
        }
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string normalize_text(const std::string& text) {
    if (text.empty()) return text;
    u32 t = decode_utf8(text);
    for (auto& c : t) {
        switch (c) {
            case U'„': case U'“': case U'«': case U'»': case U'‹': case U'›': c = U'"'; break;
            case U'‘': case U'’': c = U'\''; break;
            case U'\t': case U'\r': case 0xA0: case 0x2009: case 0x202F: c = U' '; break;
            default: break;
        }
    }
    t = expand_abbreviations(t);
    // § and §§ before a number
    {
        u32 r;
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] == U'§') {
                if (i + 1 < t.size() && t[i + 1] == U'§') { r += decode_utf8("Paragrafen "); ++i; }
                else r += decode_utf8("Paragraf ");
            } else {
                r.push_back(t[i]);
            }
        }
        t = r;
    }
    t = normalize_numbers(t);
    {
        u32 r;
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] == U'%') {
                if (!r.empty() && r.back() != U' ') r.push_back(U' ');
                r += decode_utf8("Prozent");
            } else {
                r.push_back(t[i]);
            }
        }
        t = r;
    }
    // collapse spaces, trim
    u32 r;
    for (char32_t c : t) {
        if (c == U' ' && !r.empty() && r.back() == U' ') continue;
        r.push_back(c);
    }
    while (!r.empty() && r.back() == U' ') r.pop_back();
    size_t start = 0;
    while (start < r.size() && r[start] == U' ') ++start;
    return encode_utf8(r.substr(start));
}

std::string g2p_word(const std::string& word) {
    return g2p_lower(fold_accents(lower(decode_utf8(word))), 0);
}

std::string phonemize_word(const std::string& word, const Dictionary& dict) {
    u32 w = decode_utf8(word);
    if (w.empty()) return "";
    u32 lw = lower(w);
    // Homographs whose capitalised (noun) reading differs: "Weg" vs "weg".
    if (is_upper(w[0]) && lw == U"weg") return "vˈeːk";
    if (const std::string* ps = dict_find(dict, lw)) return *ps;
    u32 folded = fold_accents(lw);
    if (folded != lw) {
        if (const std::string* ps = dict_find(dict, folded)) return *ps;
    }
    std::string out;
    if (inflected(folded, dict, out)) return out;
    if (compound(folded, dict, out, 0)) return out;
    return g2p_lower(folded, 0);
}

std::string phonemize(const std::string& text, const Dictionary& dict) {
    u32 t = decode_utf8(text);
    std::string out;
    bool pending_space = false;
    const size_t n = t.size();
    size_t i = 0;
    while (i < n) {
        char32_t c = t[i];
        if (c == U' ' || c == U'\n' || c == U'\t' || c == U'\r') {
            pending_space = true;
            ++i;
            continue;
        }
        if (is_letter(c) || c == U'\'') {
            size_t j = i;
            while (j < n && (is_letter(t[j]) || t[j] == U'\'' ||
                             (t[j] == U'-' && j + 1 < n && is_letter(t[j + 1])))) {
                ++j;
            }
            if (pending_space && !out.empty()) out += " ";
            pending_space = false;
            // Hyphenated words: each part on its own, later parts demoted.
            u32 token = t.substr(i, j - i);
            u32 part;
            bool first = true;
            auto flush = [&]() {
                u32 clean;
                for (char32_t ch : part) if (ch != U'\'') clean.push_back(ch);
                if (!clean.empty()) {
                    std::string pp = phonemize_word(encode_utf8(clean), dict);
                    out += first ? pp : demote_primary(pp);
                    first = false;
                }
                part.clear();
            };
            for (char32_t ch : token) {
                if (ch == U'-') flush();
                else part.push_back(ch);
            }
            flush();
            i = j;
            continue;
        }
        switch (c) {
            case U',': case U'.': case U'!': case U'?': case U';': case U':': case U'-':
            case U'"': case U'(': case U')':
                append_utf8(out, c);
                pending_space = false;
                break;
            default:
                break;
        }
        ++i;
    }
    return out;
}

}  // namespace speech_core::german
