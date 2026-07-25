#include "speech_core/models/moss_prompt_processor.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    using speech_core::MossPromptProcessor;
    using speech_core::MossTokenizerDecoder;

    const auto prompt = MossPromptProcessor::prepare(375);
    assert(prompt.audio_placeholder_count == 375);
    assert(prompt.input_ids.size() == 472);
    assert(prompt.eos_token_id == 151645);

    const auto span = MossPromptProcessor::make_audio_span(375);
    assert(span.size() == 386);
    assert(std::count(span.begin(), span.end(), 151671) == 375);
    assert(span[62] == 20);
    assert(span[125] == 16 && span[126] == 15);
    assert(span[189] == 16 && span[190] == 20);
    assert(span[253] == 17 && span[254] == 15);
    assert(span[317] == 17 && span[318] == 20);
    assert(span[381] == 18 && span[382] == 15);

    const auto fixture = std::filesystem::temp_directory_path()
        / "speech-core-moss-vocab-fixture.json";
    {
        std::ofstream output(fixture, std::ios::binary);
        output << R"({"Hello":0,"\u0120world":1,"!":2})";
    }
    const MossTokenizerDecoder decoder(fixture.string());
    assert(decoder.vocab_size() == 3);
    assert(decoder.decode({0, 1, 2, 151645}) == "Hello world!");
    std::filesystem::remove(fixture);

    std::cout << "MOSS prompt/tokenizer tests passed\n";
    return 0;
}
