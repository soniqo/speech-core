// German Kokoro front-end: text normalization, rule-based G2P, dictionary
// fallbacks and the phonemizer routing. Pure C++ — no model runtime.
//
// Expected phoneme strings follow espeak-ng 1.52 German as post-processed by
// misaki (the kikiri-tts training front-end): I/W for ai/au, ʦ/ʧ/ʤ
// affricates, ɜ for a vocalic -er, the short ü written y, stress marks
// before the stressed vowel.

#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/models/kokoro_german.h"
#include "speech_core/models/kokoro_multilingual.h"
#include "speech_core/models/kokoro_phonemizer.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <unordered_map>

using speech_core::german::Dictionary;

namespace {

int failures = 0;

void expect_eq(const char* what, const std::string& got, const std::string& want) {
    if (got != want) {
        ++failures;
        std::printf("  FAIL %s: got \"%s\" want \"%s\"\n", what, got.c_str(), want.c_str());
    }
}

void test_normalize_numbers() {
    std::printf("  test_normalize_numbers ... ");
    const int before = failures;
    using speech_core::german::normalize_text;
    expect_eq("integer", normalize_text("Ich habe 21 Katzen."), "Ich habe einundzwanzig Katzen.");
    expect_eq("one", normalize_text("1 Hund"), "eins Hund");
    expect_eq("decimal", normalize_text("3,14 ist Pi"), "drei Komma eins vier ist Pi");
    expect_eq("grouped", normalize_text("1.234.567 Euro"),
              "eine Million zweihundertvierunddreißigtausendfünfhundertsiebenundsechzig Euro");
    expect_eq("time", normalize_text("um 18:20 Uhr"), "um achtzehn Uhr zwanzig");
    expect_eq("time-plain", normalize_text("gegen 9:05"), "gegen neun Uhr fünf");
    expect_eq("date", normalize_text("am 14.05.2026"), "am vierzehnten Mai zweitausendsechsundzwanzig");
    expect_eq("date-nominative", normalize_text("Heute ist der 3.10.1990."),
              "Heute ist der dritte Oktober neunzehnhundertneunzig.");
    expect_eq("ordinal", normalize_text("die 2. Schiene"), "die zweite Schiene");
    expect_eq("ordinal-am", normalize_text("am 1. Mai"), "am ersten Mai");
    expect_eq("year", normalize_text("seit 1999"), "seit neunzehnhundertneunundneunzig");
    expect_eq("currency-after", normalize_text("12,80 €"), "zwölf Euro und achtzig Cent");
    expect_eq("currency-before", normalize_text("$ 5"), "fünf Dollar");
    expect_eq("percent", normalize_text("25 % mehr"), "fünfundzwanzig Prozent mehr");
    expect_eq("unit", normalize_text("1,5 kg und 500 g"), "eins Komma fünf Kilogramm und fünfhundert Gramm");
    expect_eq("phone", normalize_text("Ruf 0170 123456 an"),
              "Ruf null eins sieben null eins zwei drei vier fünf sechs an");
    expect_eq("abbreviation", normalize_text("Prof. Klein, z.B. heute"), "Professor Klein, zum Beispiel heute");
    expect_eq("abbreviation-digit", normalize_text("Nr. 7"), "Nummer sieben");
    expect_eq("quotes", normalize_text("„Hallo“ sagte er"), "\"Hallo\" sagte er");
    expect_eq("negative-unchanged", normalize_text("Hallo Welt"), "Hallo Welt");
    std::printf(failures != before ? "FAIL\n" : "ok\n");
}

void test_g2p_rules() {
    std::printf("  test_g2p_rules ... ");
    const int before = failures;
    using speech_core::german::g2p_word;
    struct Case { const char* word; const char* ipa; };
    const Case cases[] = {
        // vowel length and quality
        {"Tag", "tˈɑːk"}, {"Wasser", "vˈasɜ"}, {"Nase", "nˈɑːzə"}, {"Fenster", "fˈɛnstɜ"},
        {"Küche", "kˈyçə"}, {"Kuchen", "kˈuːxən"}, {"Mädchen", "mˈɛːtçən"}, {"Vögel", "fˈøːɡəl"},
        {"Schule", "ʃˈuːlə"}, {"Straße", "ʃtɾˈɑːsə"}, {"Fahrrad", "fˈɑːrɑːt"},
        // diphthongs and umlauts
        {"Zeit", "ʦˈIt"}, {"Haus", "hˈWs"}, {"Leute", "lˈɔøtə"}, {"Häuser", "hˈɔøzɜ"},
        // consonant rules
        {"sechs", "zˈɛks"}, {"Nacht", "nˈaxt"}, {"Licht", "lˈɪçt"}, {"Chemie", "çeːmˈiː"},
        {"Quelle", "kvˈɛlə"}, {"Pfütze", "pfˈyʦə"}, {"Sport", "ʃpˈɔɾt"}, {"Stein", "ʃtˈIn"},
        {"Vater", "fˈɑːtɜ"}, {"Klavier", "klavˈiːɾ"}, {"Freund", "frˈɔønt"}, {"Brot", "bɾˈoːt"},
        {"Burg", "bˈʊɐk"}, {"Kurs", "kˈuːɾs"}, {"deutsch", "dˈɔøtʃ"}, {"Tschüss", "ʧˈys"},
        {"lebend", "lˈeːbənt"}, {"Kindern", "kˈɪndɜn"}, {"redete", "rˈeːdətə"},
        // prefixes and suffixes
        {"verstehen", "fɛɾʃtˈeːən"}, {"Gebäude", "ɡəbˈɔødə"}, {"anfangen", "ˈanfˌaŋən"},
        {"Universität", "ˌuːniːveːɾziːtˈɛːt"}, {"Information", "ˌɪnfɔɾmaʦjˈoːn"},
        {"studieren", "ʃtuːdˈiːrən"}, {"Wissenschaft", "vˈɪsənʃˌaft"}, {"Lehrerin", "lˈeːrərˌɪn"},
        {"Beratung", "bərˈɑːtʊŋ"}, {"König", "kˈøːnɪç"}, {"Freundlichkeit", "frˈɔøntlɪçkˌIt"},
        {"ausgewählt", "ˈWsɡəvˌɛːlt"}, {"bevorstehen", "bəfˈoːɾʃtˌeːən"}, {"herunter", "hɛɾˈʊntɜ"},
        // case is folded
        {"TAG", "tˈɑːk"},
    };
    for (const auto& c : cases) expect_eq(c.word, g2p_word(c.word), c.ipa);
    expect_eq("empty", g2p_word(""), "");
    std::printf(failures != before ? "FAIL\n" : "ok\n");
}

void test_dictionary_fallbacks() {
    std::printf("  test_dictionary_fallbacks ... ");
    const int before = failures;
    using speech_core::german::phonemize_word;
    Dictionary dict = {
        {"kind", "kˈɪnt"}, {"kinder", "kˈɪndɜ"}, {"tag", "tˈɑːk"}, {"haus", "hˈWs"},
        {"sagen", "zˈɑːɡən"}, {"kritisieren", "kɾˌiːtiːzˈiːrən"}, {"willig", "vˈɪlɪç"},
        {"eisen", "ˈIzən"}, {"bahn", "bˈɑːn"}, {"hof", "hˈoːf"}, {"kleider", "klˈIdɜ"},
        {"ordnung", "ˈɔɾdnʊŋ"}, {"weg", "vɛk"}, {"geringer", "ɡərˈɪŋɜ"}, {"cafe", "kafˈeː"},
        {"leben", "lˈeːbən"},
    };
    expect_eq("direct", phonemize_word("Kind", dict), "kˈɪnt");
    expect_eq("plural-n", phonemize_word("Kindern", dict), "kˈɪndɜn");
    expect_eq("revoice", phonemize_word("Tage", dict), "tˈɑːɡə");
    expect_eq("genitive", phonemize_word("Tages", dict), "tˈɑːɡəs");
    expect_eq("verb-t", phonemize_word("sagt", dict), "zˈɑːkt");
    expect_eq("verb-te", phonemize_word("lebte", dict), "lˈeːptə");
    expect_eq("verb-st-r", phonemize_word("kritisierst", dict), "kɾˌiːtiːzˈiːɾst");
    expect_eq("ig-vowel", phonemize_word("willigen", dict), "vˈɪlɪɡən");
    expect_eq("er-vowel", phonemize_word("geringere", dict), "ɡərˈɪŋərə");
    expect_eq("compound", phonemize_word("Eisenbahn", dict), "ˈIzənbˌɑːn");
    expect_eq("compound-2syl", phonemize_word("Bahnhof", dict), "bˈɑːnhoːf");
    expect_eq("compound-er", phonemize_word("Kleiderordnung", dict), "klˈIdərˌɔɾdnʊŋ");
    expect_eq("compound-link-s", phonemize_word("Kindeshaus", dict), "kˈɪndəshˌWs");
    expect_eq("rules-fallback", phonemize_word("Zwerge", dict), "ʦvˈɛɾɡə");
    expect_eq("homograph-noun", phonemize_word("Weg", dict), "vˈeːk");
    expect_eq("homograph-adverb", phonemize_word("weg", dict), "vɛk");
    expect_eq("accent-folded-lookup", phonemize_word("Café", dict), "kafˈeː");
    std::printf(failures != before ? "FAIL\n" : "ok\n");
}

void test_phonemize_text() {
    std::printf("  test_phonemize_text ... ");
    const int before = failures;
    Dictionary dict = {{"guten", "ɡˈuːtən"}, {"tag", "tˈɑːk"}, {"wie", "viː"}, {"geht", "ɡˈeːt"},
                       {"es", "ɛs"}, {"ihnen", "iːnən"}, {"mein", "mIn"}, {"name", "nˈɑːmə"},
                       {"ist", "ɪst"}, {"um", "ʊm"}, {"uhr", "ˈuːɾ"}, {"acht", "ˈaxt"}};
    const std::string got = speech_core::multilingual::german_phonemize(
        "Guten Tag, wie geht es Ihnen? Mein Name ist Martin.", dict);
    expect_eq("sentence", got, "ɡˈuːtən tˈɑːk, viː ɡˈeːt ɛs iːnən? mIn nˈɑːmə ɪst mˈaɾtɪn.");
    expect_eq("normalized", speech_core::multilingual::german_phonemize("um 8 Uhr", dict), "ʊm ˈaxt ˈuːɾ");
    expect_eq("hyphen", speech_core::german::phonemize("E-Mail", dict), "ˈeːmˌIl");
    expect_eq("empty", speech_core::multilingual::german_phonemize("", dict), "");
    std::printf(failures != before ? "FAIL\n" : "ok\n");
}

void test_phonemizer_routing() {
    std::printf("  test_phonemizer_routing ... ");
    const int before = failures;
    speech_core::KokoroPhonemizer phonemizer;
    phonemizer.set_language("de");
    // No dictionary loaded: the rule-based path still produces German
    // phonemes, with the r/ɾ convention untouched by the Romance post-processing.
    const std::string got = phonemizer.text_to_phonemes("Straße");
    expect_eq("routing", got, "ʃtɾˈɑːsə");
    std::printf(failures != before ? "FAIL\n" : "ok\n");
}

}  // namespace

int main() {
    std::printf("test_kokoro_german\n");
    test_normalize_numbers();
    test_g2p_rules();
    test_dictionary_fallbacks();
    test_phonemize_text();
    test_phonemizer_routing();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
