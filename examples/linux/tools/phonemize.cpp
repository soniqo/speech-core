// Tiny CLI that dumps the phoneme string + token IDs the Kokoro phonemizer
// produces for a piece of text. Used to verify text→phoneme conversion is
// correct before blaming the TTS model.
//
// Usage: speech_phonemize [model_dir] "<text>" [language]

#include <speech_core/models/kokoro_phonemizer.h>

#include "../../common/default_model_dir.h"

#include <cstdio>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s [model_dir] \"<text>\" [language]\n"
            "  model_dir : directory holding vocab_index.json + dictionaries\n"
            "              (default: $SPEECH_MODEL_DIR, else %s)\n"
            "  language  : BCP-47 tag (default: en)\n",
            argv[0], speech_example_model_dir().c_str());
        return 2;
    }
    // model_dir is optional. With 3 args, <model_dir> <text> and
    // <text> <language> are both plausible — disambiguate by whether argv[1]
    // is an existing directory.
    const bool has_dir = (argc >= 4)
        || (argc == 3 && std::filesystem::is_directory(argv[1]));
    const int base = has_dir ? 2 : 1;
    const std::string model_dir = has_dir ? argv[1] : speech_example_model_dir();
    const std::string text      = argv[base];
    const std::string language  = (argc >= base + 2) ? argv[base + 1] : "en";

    speech_core::KokoroPhonemizer p;
    if (!p.load_vocab(model_dir + "/vocab_index.json")) {
        std::fprintf(stderr, "failed to load vocab from %s/vocab_index.json\n",
                     model_dir.c_str());
        return 1;
    }
    p.load_dictionaries(model_dir);
    for (const char* lang : {"fr", "es", "it", "pt", "hi"}) {
        p.load_language_dict(lang, model_dir + "/dict_" + std::string(lang) + ".json");
    }
    p.set_language(language);

    std::string phonemes = p.text_to_phonemes(text);
    auto ids = p.tokenize(text, 128);

    std::printf("text     : %s\n", text.c_str());
    std::printf("language : %s\n", language.c_str());
    std::printf("phonemes : %s\n", phonemes.c_str());
    std::printf("tokens   : [%zu]", ids.size());
    for (auto id : ids) std::printf(" %lld", static_cast<long long>(id));
    std::printf("\n");
    return 0;
}
