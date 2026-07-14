#include "speech_core/models/pocket_tts_tokenizer.h"

#include "nlohmann/json.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace speech_core {

namespace {

// The dynamic-programming tokenization scheme is adapted from sherpa-onnx's
// SentencePieceTokenizer (Apache-2.0, Copyright 2026 Xiaomi Corporation).
// It consumes the same vocab.json/token_scores.json representation produced
// by the Pocket TTS ONNX exporter.
constexpr float kNegativeInfinity = -1.0e30f;

nlohmann::json load_json(const std::string& filename) {
    if (filename.empty()) {
        throw std::invalid_argument("Pocket TTS tokenizer JSON path is empty");
    }
    std::ifstream stream(filename);
    if (!stream) {
        throw std::runtime_error("Cannot open Pocket TTS tokenizer file: " + filename);
    }
    nlohmann::json value;
    stream >> value;
    if (!value.is_object()) {
        throw std::runtime_error("Pocket TTS tokenizer file is not a JSON object: " + filename);
    }
    return value;
}

}  // namespace

class PocketTtsTokenizer::Impl {
public:
    Impl(const std::string& vocab_json, const std::string& token_scores_json) {
        const auto vocabulary = load_json(vocab_json);
        const auto scores = load_json(token_scores_json);
        if (vocabulary.size() != scores.size()) {
            throw std::runtime_error("Pocket TTS vocabulary and score table sizes differ");
        }

        token_to_id_.reserve(vocabulary.size());
        id_to_token_.resize(vocabulary.size());
        for (const auto& item : vocabulary.items()) {
            const auto id = item.value().get<std::int32_t>();
            if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
                throw std::runtime_error("Pocket TTS vocabulary contains an out-of-range token ID");
            }
            token_to_id_.emplace(item.key(), id);
            id_to_token_[static_cast<std::size_t>(id)] = item.key();
        }

        token_to_score_.reserve(scores.size());
        for (const auto& item : scores.items()) {
            token_to_score_.emplace(item.key(), item.value().get<float>());
        }

        byte_token_id_.fill(-1);
        byte_token_score_.fill(kNegativeInfinity);
        build_trie();
    }

    std::vector<std::int32_t> encode_ids(const std::string& text) const {
        std::vector<std::int32_t> ids;
        encode(text, &ids, nullptr);
        return ids;
    }

    std::vector<std::string> encode_tokens(const std::string& text) const {
        std::vector<std::string> tokens;
        encode(text, nullptr, &tokens);
        return tokens;
    }

private:
    struct TrieNode {
        std::unordered_map<unsigned char, std::int32_t> next;
        std::int32_t token_id = -1;
        float score = 0.0f;
    };

    void build_trie() {
        trie_.reserve(token_to_id_.size() * 2);
        trie_.emplace_back();

        for (const auto& entry : token_to_id_) {
            const std::string& token = entry.first;
            std::int32_t node = 0;
            for (const unsigned char byte : token) {
                auto next = trie_[static_cast<std::size_t>(node)].next.find(byte);
                if (next == trie_[static_cast<std::size_t>(node)].next.end()) {
                    const auto new_node = static_cast<std::int32_t>(trie_.size());
                    trie_[static_cast<std::size_t>(node)].next.emplace(byte, new_node);
                    trie_.emplace_back();
                    node = new_node;
                } else {
                    node = next->second;
                }
            }
            const auto score = token_to_score_.find(token);
            if (score == token_to_score_.end()) {
                throw std::runtime_error("Pocket TTS token is missing its score: " + token);
            }
            auto& leaf = trie_[static_cast<std::size_t>(node)];
            leaf.token_id = entry.second;
            leaf.score = score->second;
        }

        for (std::int32_t byte = 0; byte < 256; ++byte) {
            char name[8];
            std::snprintf(name, sizeof(name), "<0x%02X>", byte);
            const auto id = token_to_id_.find(name);
            const auto score = token_to_score_.find(name);
            if (id != token_to_id_.end() && score != token_to_score_.end()) {
                byte_token_id_[static_cast<std::size_t>(byte)] = id->second;
                byte_token_score_[static_cast<std::size_t>(byte)] = score->second;
            }
        }
    }

    void encode(const std::string& input,
                std::vector<std::int32_t>* ids,
                std::vector<std::string>* tokens) const {
        std::string text;
        text.reserve(input.size() + 8);
        for (const char character : input) {
            if (character == ' ') {
                text.append("\xE2\x96\x81");  // SentencePiece whitespace marker: U+2581.
            } else {
                text.push_back(character);
            }
        }
        if (text.rfind("\xE2\x96\x81", 0) != 0) {
            text.insert(0, "\xE2\x96\x81");
        }

        const auto length = static_cast<std::int32_t>(text.size());
        std::vector<float> best(static_cast<std::size_t>(length + 1), kNegativeInfinity);
        std::vector<std::int32_t> back(static_cast<std::size_t>(length + 1), -1);
        std::vector<std::int32_t> back_id(static_cast<std::size_t>(length + 1), -1);
        best[static_cast<std::size_t>(length)] = 0.0f;

        for (std::int32_t start = length - 1; start >= 0; --start) {
            std::int32_t node = 0;
            for (std::int32_t end = start; end < length; ++end) {
                const auto byte = static_cast<unsigned char>(text[static_cast<std::size_t>(end)]);
                const auto next = trie_[static_cast<std::size_t>(node)].next.find(byte);
                if (next == trie_[static_cast<std::size_t>(node)].next.end()) {
                    break;
                }
                node = next->second;
                const auto& candidate = trie_[static_cast<std::size_t>(node)];
                if (candidate.token_id >= 0) {
                    const float score = candidate.score + best[static_cast<std::size_t>(end + 1)];
                    if (score > best[static_cast<std::size_t>(start)]) {
                        best[static_cast<std::size_t>(start)] = score;
                        back[static_cast<std::size_t>(start)] = end + 1;
                        back_id[static_cast<std::size_t>(start)] = candidate.token_id;
                    }
                }
            }

            if (back[static_cast<std::size_t>(start)] < 0) {
                const auto byte = static_cast<unsigned char>(text[static_cast<std::size_t>(start)]);
                const auto fallback_id = byte_token_id_[byte];
                if (fallback_id >= 0) {
                    best[static_cast<std::size_t>(start)] =
                        byte_token_score_[byte] + best[static_cast<std::size_t>(start + 1)];
                    back_id[static_cast<std::size_t>(start)] = fallback_id;
                }
                back[static_cast<std::size_t>(start)] = start + 1;
            }
        }

        for (std::int32_t offset = 0; offset < length;) {
            const auto next = back[static_cast<std::size_t>(offset)];
            const auto id = back_id[static_cast<std::size_t>(offset)];
            if (next <= offset || id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
                throw std::runtime_error("Pocket TTS tokenizer could not reconstruct its best path");
            }
            if (ids) ids->push_back(id);
            if (tokens) tokens->push_back(id_to_token_[static_cast<std::size_t>(id)]);
            offset = next;
        }
    }

    std::vector<TrieNode> trie_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::int32_t> token_to_id_;
    std::unordered_map<std::string, float> token_to_score_;
    std::array<std::int32_t, 256> byte_token_id_{};
    std::array<float, 256> byte_token_score_{};
};

PocketTtsTokenizer::PocketTtsTokenizer(const std::string& vocab_json,
                                       const std::string& token_scores_json)
    : impl_(std::make_unique<Impl>(vocab_json, token_scores_json)) {}

PocketTtsTokenizer::~PocketTtsTokenizer() = default;
PocketTtsTokenizer::PocketTtsTokenizer(PocketTtsTokenizer&&) noexcept = default;
PocketTtsTokenizer& PocketTtsTokenizer::operator=(PocketTtsTokenizer&&) noexcept = default;

std::vector<std::int32_t> PocketTtsTokenizer::encode_ids(const std::string& text) const {
    return impl_->encode_ids(text);
}

std::vector<std::string> PocketTtsTokenizer::encode_tokens(const std::string& text) const {
    return impl_->encode_tokens(text);
}

}  // namespace speech_core
