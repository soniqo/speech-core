#include "speech_core/models/onnx_sortformer_diarizer.h"

#include "speech_core/audio/mel.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace speech_core {
namespace {

/// Read a tensor's shape from the session, by index, for inputs or outputs.
std::vector<std::int64_t> tensor_shape(
    const OrtApi* api, OrtSession* session, std::size_t index, bool input) {
    OrtTypeInfo* info = nullptr;
    ort_check(api, input
        ? api->SessionGetInputTypeInfo(session, index, &info)
        : api->SessionGetOutputTypeInfo(session, index, &info));
    const OrtTensorTypeAndShapeInfo* shape = nullptr;
    ort_check(api, api->CastTypeInfoToTensorInfo(info, &shape));
    std::size_t dims = 0;
    api->GetDimensionsCount(shape, &dims);
    std::vector<std::int64_t> out(dims);
    api->GetDimensions(shape, out.data(), dims);
    api->ReleaseTypeInfo(info);
    return out;
}

[[noreturn]] void wrong_shape(const std::string& what) {
    throw std::runtime_error(
        "Sortformer ONNX: unexpected " + what
        + ". The geometry comes from the graph and the windowing from the "
          "bundle's config.json; a graph and a config describing different "
          "variants would otherwise be driven with the wrong chunk length and "
          "answer confidently rather than fail.");
}

/// The graph's mel front-end, settled by diffing against features dumped from
/// the checkpoint's own preprocessor rather than read off its config. Two of
/// these are invisible in the config: `torch.stft` centres the 400-sample
/// window inside the 512-point FFT, and NeMo pads constant rather than
/// reflect. Guessing either gives features the model answers confidently and
/// wrongly, with nothing in the pipeline to notice.
constexpr bool kSlaneyNorm = true;
constexpr bool kCenter = true;
constexpr bool kTorchStftLayout = true;
constexpr bool kCenterPadZeros = true;
constexpr bool kSymmetricWindow = true;
constexpr float kPreEmphasis = 0.97f;

/// `log(x + 2^-24)`, which the reference's own minimum confirms at -16.6355.
const float kLogFloor = std::ldexp(1.0f, -24);

/// Mel frames of real audio needed either side of a window so that centring is
/// exact rather than padded: a frame spans n_fft/2 = 256 samples each way,
/// which is 1.6 hops.
constexpr std::int64_t kCentringMargin = 2;

/// Read one positive integer from a bundle's `config.json`, if it says so.
///
/// Absent leaves the caller's value alone: the first published bundle predates
/// this and carries the `default` variant's numbers, which are the defaults.
/// Present but unparseable or non-positive is a fault rather than a fallback —
/// a bundle that describes itself wrongly is exactly the case this exists to
/// catch, and continuing with a default would restore the silent failure.
bool config_int(
    const json::Dict& config, const char* key, int& out,
    const std::string& where) {
    const auto found = config.find(key);
    if (found == config.end()) return false;
    int value = 0;
    try {
        value = std::stoi(found->second);
    } catch (...) {
        throw std::runtime_error(
            std::string("Sortformer bundle ") + where + " has a non-numeric "
            + key);
    }
    if (value <= 0) {
        throw std::runtime_error(
            std::string("Sortformer bundle ") + where + " has a non-positive "
            + key);
    }
    out = value;
    return true;
}

}  // namespace

OnnxSortformerDiarizer::OnnxSortformerDiarizer(
    const std::string& model_path, bool hw_accel) {
    api_ = OnnxEngine::get().api();

    // Three numbers decide how this graph is driven and none of them is in it.
    // `chunk_right_context` sets how many of a window's frames belong to the
    // next call, so getting it wrong advances the timeline by the wrong amount
    // every call and every label lands at the wrong time. `spkcache_update_
    // period` decides which frames the arrival-order cache evicts, and pairing
    // one variant's graph with another's period evicts the wrong ones while
    // looking healthy — the model still emits four plausible probabilities per
    // frame and simply stops meaning the same person.
    //
    // The defaults are the `default` variant's, which was the only bundle that
    // existed when they were written. A `balanced` graph driven by them would
    // take 121 window frames as 80 chunk frames instead of 100 and evict on a
    // 188-frame period instead of 100: two silent faults, from a file that was
    // sitting beside the model saying so. So the bundle is asked.
    int declared_chunk_len = 0;
    {
        namespace fs = std::filesystem;
        const fs::path model = model_path;
        // `sortformer-balanced.config.json` before `config.json`, because two
        // variants can share a directory and a single `config.json` cannot
        // describe both. Installing a second graph beside the first would
        // otherwise silently redescribe the first -- the precise failure this
        // whole change exists to prevent, and easy to do by accident.
        //
        // The bare name stays as the fallback so the published single-variant
        // bundle, which has one graph and one `config.json`, keeps working.
        fs::path config_path =
            model.parent_path() / (model.stem().string() + ".config.json");
        if (!fs::is_regular_file(config_path)) {
            config_path = model.parent_path() / "config.json";
        }
        if (fs::is_regular_file(config_path)) {
            const auto config = json::parse_flat_object(
                json::read_file(config_path.string()));
            const std::string where = config_path.string();
            config_int(config, "chunk_left_context", cfg_.left_context, where);
            config_int(
                config, "chunk_right_context", cfg_.right_context, where);
            config_int(config, "subsampling_factor", cfg_.subsampling, where);
            config_int(
                config, "spkcache_update_period", cfg_.cache.update_period,
                where);
            config_int(config, "chunk_len", declared_chunk_len, where);
        }
    }

    session_ = OnnxEngine::get().load(model_path, hw_accel);

    // Every dimension below is READ FROM THE GRAPH. The windowing this class
    // performs is derived from them, so a mismatch has to be a refusal here
    // rather than a subtly wrong timeline later.
    const auto chunk = tensor_shape(api_, session_, 0, true);
    const auto spkcache = tensor_shape(api_, session_, 2, true);
    const auto fifo = tensor_shape(api_, session_, 4, true);
    const auto preds = tensor_shape(api_, session_, 0, false);
    const auto embeddings = tensor_shape(api_, session_, 1, false);

    if (chunk.size() != 3 || spkcache.size() != 3 || fifo.size() != 3
        || preds.size() != 3 || embeddings.size() != 3) {
        wrong_shape("tensor rank");
    }

    window_mels_ = static_cast<int>(chunk[1]);
    cfg_.mel_bins = static_cast<int>(chunk[2]);
    cache_frames_ = static_cast<int>(spkcache[1]);
    embedding_dim_ = static_cast<int>(spkcache[2]);
    fifo_frames_ = static_cast<int>(fifo[1]);
    speakers_ = static_cast<int>(preds[2]);
    window_frames_ = static_cast<int>(embeddings[1]);

    if (window_mels_ <= 0 || window_frames_ <= 0 || speakers_ <= 0
        || cache_frames_ <= 0 || fifo_frames_ <= 0 || embedding_dim_ <= 0) {
        wrong_shape("a dynamic dimension where this export pins one");
    }
    if (window_mels_ != window_frames_ * cfg_.subsampling) {
        wrong_shape("mel frames per encoder frame");
    }
    if (fifo[2] != spkcache[2] || embeddings[2] != spkcache[2]) {
        wrong_shape("embedding width");
    }
    // The prediction block covers the cache, the FIFO and the whole window
    // including its context. Anything else means the offsets this class hands
    // the cache would land in the wrong section.
    if (preds[1] != cache_frames_ + fifo_frames_ + window_frames_) {
        wrong_shape("prediction block length");
    }

    chunk_frames_ =
        window_frames_ - cfg_.left_context - cfg_.right_context;
    if (chunk_frames_ <= 0) wrong_shape("context wider than the window");
    // The one check that catches a config paired with the wrong graph. The
    // contexts come from the file and the window comes from the model, so if
    // the file also names its chunk length, the three have to agree — and when
    // they do not, every frame this class publishes would be misdated. Both
    // sources have to be wrong in exactly compensating ways to slip through.
    if (declared_chunk_len != 0 && declared_chunk_len != chunk_frames_) {
        wrong_shape(
            "chunk length: config.json says " + std::to_string(declared_chunk_len)
            + " but the graph's window less its context is "
            + std::to_string(chunk_frames_));
    }

    cfg_.cache.speakers = speakers_;
    cfg_.cache.cache_frames = cache_frames_;
    cfg_.cache.fifo_frames = fifo_frames_;
    cfg_.cache.embedding_dim = embedding_dim_;
    cache_ = std::make_unique<SortformerSpeakerCache>(cfg_.cache);
}

OnnxSortformerDiarizer::~OnnxSortformerDiarizer() {
    if (session_) api_->ReleaseSession(session_);
}

float OnnxSortformerDiarizer::frame_seconds() const {
    return static_cast<float>(cfg_.subsampling)
        * static_cast<float>(cfg_.mel_hop)
        / static_cast<float>(cfg_.sample_rate);
}

void OnnxSortformerDiarizer::reset() {
    cache_->reset();
    audio_.clear();
    audio_start_sample_ = 0;
    samples_seen_ = 0;
    step_ = 0;
    frames_emitted_ = 0;
    finished_ = false;
}

std::vector<float> OnnxSortformerDiarizer::window_features(
    std::int64_t first_frame) const {
    const std::int64_t bins = cfg_.mel_bins;
    const std::int64_t hop = cfg_.mel_hop;
    std::vector<float> window(
        static_cast<std::size_t>(window_mels_) * static_cast<std::size_t>(bins),
        0.0f);

    // Mel frames this window covers, and the part of them the recording
    // actually has. A window reaches before the first sample on the first step
    // and past the last on the final one; both are zeros, which is what a whole
    // -recording front-end would produce there too.
    const std::int64_t want_lo = first_frame * cfg_.subsampling;
    const std::int64_t want_hi = want_lo + window_mels_;
    const std::int64_t have = static_cast<std::int64_t>(samples_seen_ / hop);
    const std::int64_t lo = std::max<std::int64_t>(want_lo, 0);
    const std::int64_t hi = std::min<std::int64_t>(want_hi, have);
    if (hi <= lo) return window;

    // Hand the front-end a margin of real audio either side so its own centring
    // padding never stands in for samples the recording has.
    const std::int64_t from = std::max<std::int64_t>(lo - kCentringMargin, 0) * hop;
    const std::int64_t to = std::min<std::int64_t>(
        (hi + kCentringMargin) * hop, static_cast<std::int64_t>(samples_seen_));
    if (to <= from) return window;

    const std::size_t offset =
        static_cast<std::size_t>(from) - audio_start_sample_;
    const std::size_t count = static_cast<std::size_t>(to - from);
    if (offset + count > audio_.size()) return window;

    // Pre-emphasis is the caller's job for this front-end, as it is for the
    // other NeMo-derived wrappers here. Its one-sample memory is taken from the
    // real previous sample where there is one, so a window boundary does not
    // become a discontinuity.
    std::vector<float> pcm(audio_.begin() + offset,
                           audio_.begin() + offset + count);
    float previous = offset > 0 ? audio_[offset - 1] : 0.0f;
    for (std::size_t index = 0; index < pcm.size(); ++index) {
        const float current = pcm[index];
        pcm[index] = current - kPreEmphasis * previous;
        previous = current;
    }

    const std::vector<float> mel = audio::mel_spectrogram(
        pcm.data(), pcm.size(), cfg_.sample_rate, cfg_.n_fft, cfg_.mel_hop,
        cfg_.win_length, cfg_.mel_bins, kSlaneyNorm, kLogFloor, kCenter,
        kTorchStftLayout, kCenterPadZeros, kSymmetricWindow);
    if (mel.empty()) return window;
    const std::int64_t produced =
        static_cast<std::int64_t>(mel.size()) / bins;
    const std::int64_t base = from / hop;

    // mel_spectrogram is [bins, frames]; the graph wants [frames, bins].
    for (std::int64_t frame = lo; frame < hi; ++frame) {
        const std::int64_t source = frame - base;
        if (source < 0 || source >= produced) continue;
        const std::int64_t slot = frame - want_lo;
        for (std::int64_t bin = 0; bin < bins; ++bin) {
            window[static_cast<std::size_t>(slot * bins + bin)] =
                mel[static_cast<std::size_t>(bin * produced + source)];
        }
    }
    return window;
}

bool OnnxSortformerDiarizer::advance_step(
    bool flush, std::vector<float>& out) {
    const std::int64_t hop = cfg_.mel_hop;
    const std::int64_t first_frame =
        static_cast<std::int64_t>(step_) * chunk_frames_ - cfg_.left_context;
    const std::int64_t want_hi =
        (first_frame + window_frames_) * cfg_.subsampling;

    if (!flush) {
        // Wait until the whole window, plus the centring margin, has arrived.
        const std::int64_t needed = (want_hi + kCentringMargin) * hop;
        if (static_cast<std::int64_t>(samples_seen_) < needed) return false;
    } else {
        // On the final pass, only steps that own real audio are worth running.
        const std::int64_t have =
            static_cast<std::int64_t>(samples_seen_ / hop) / cfg_.subsampling;
        if (static_cast<std::int64_t>(step_) * chunk_frames_ >= have) {
            return false;
        }
    }

    const std::vector<float> window = window_features(first_frame);

    OrtMemoryInfo* memory = nullptr;
    ort_check(api_, api_->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &memory));

    // The graph tensors have fixed capacities, but their length inputs describe
    // the valid prefixes. NeMo starts both prefixes at zero; treating the
    // initial zero padding as 188 valid cache frames changes attention on the
    // first call and poisons the predictions used to seed arrival order.
    std::vector<float> spkcache(
        static_cast<std::size_t>(cache_frames_) * embedding_dim_, 0.0f);
    std::copy(cache_->cache().begin(), cache_->cache().end(), spkcache.begin());
    std::vector<float> fifo(
        static_cast<std::size_t>(fifo_frames_) * embedding_dim_, 0.0f);
    std::copy(cache_->fifo().begin(), cache_->fifo().end(), fifo.begin());

    const std::int64_t held = static_cast<std::int64_t>(cache_->fifo_frames());
    std::int64_t chunk_length = window_mels_;
    std::int64_t spkcache_length =
        static_cast<std::int64_t>(cache_->cache_frames());
    std::int64_t fifo_length = held;

    const std::int64_t chunk_dims[3] = {1, window_mels_, cfg_.mel_bins};
    const std::int64_t cache_dims[3] = {1, cache_frames_, embedding_dim_};
    const std::int64_t fifo_dims[3] = {1, fifo_frames_, embedding_dim_};
    const std::int64_t scalar_dims[1] = {1};

    OrtValue* inputs[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    auto make_float = [&](int slot, std::vector<float>& data,
                          const std::int64_t* dims, std::size_t rank) {
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            memory, data.data(), data.size() * sizeof(float), dims, rank,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[slot]));
    };
    auto make_int = [&](int slot, std::int64_t& value) {
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            memory, &value, sizeof(value), scalar_dims, 1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[slot]));
    };
    std::vector<float> mutable_window = window;
    make_float(0, mutable_window, chunk_dims, 3);
    make_int(1, chunk_length);
    make_float(2, spkcache, cache_dims, 3);
    make_int(3, spkcache_length);
    make_float(4, fifo, fifo_dims, 3);
    make_int(5, fifo_length);

    const char* input_names[6] = {
        "chunk", "chunk_lengths", "spkcache", "spkcache_lengths",
        "fifo", "fifo_lengths"};
    const char* output_names[3] = {
        "spkcache_fifo_chunk_preds", "chunk_pre_encode_embs",
        "chunk_pre_encode_lengths"};
    OrtValue* outputs[3] = {nullptr, nullptr, nullptr};

    OrtStatus* status = api_->Run(
        session_, nullptr, input_names, inputs, 6, output_names, 3, outputs);

    float* predictions = nullptr;
    float* embeddings = nullptr;
    if (status == nullptr) {
        api_->GetTensorMutableData(outputs[0],
                                   reinterpret_cast<void**>(&predictions));
        api_->GetTensorMutableData(outputs[1],
                                   reinterpret_cast<void**>(&embeddings));
    }

    std::vector<float> chunk_predictions;
    if (predictions != nullptr && embeddings != nullptr) {
        // The graph's fixed_concat_and_pad wrapper packs the valid cache, FIFO,
        // and chunk prefixes at the front of its static output tensor. On a
        // fresh recording that puts the chunk at frame zero, not after the
        // cache/FIFO capacities. Reading it as fixed blocks shifts the first
        // chunk by the 40-frame FIFO width and mismatches cache embeddings with
        // the probabilities that score them.
        chunk_predictions = cache_->advance(
            embeddings, static_cast<std::size_t>(window_frames_),
            predictions,
            static_cast<std::size_t>(
                cache_frames_ + fifo_frames_ + window_frames_),
            cfg_.left_context, cfg_.right_context,
            SortformerSpeakerCache::PredictionLayout::Packed);
    }

    for (OrtValue* value : outputs) if (value) api_->ReleaseValue(value);
    for (OrtValue* value : inputs) if (value) api_->ReleaseValue(value);
    api_->ReleaseMemoryInfo(memory);
    if (status != nullptr) {
        ort_check(api_, status);
    }

    out.insert(out.end(), chunk_predictions.begin(), chunk_predictions.end());
    frames_emitted_ += chunk_predictions.size()
        / static_cast<std::size_t>(speakers_);
    ++step_;

    // Audio before the next window's margin can go. A recording is hours long
    // and a window is half a minute of it.
    const std::int64_t next_first =
        static_cast<std::int64_t>(step_) * chunk_frames_ - cfg_.left_context;
    const std::int64_t keep_from = std::max<std::int64_t>(
        (next_first * cfg_.subsampling - kCentringMargin) * hop, 0);
    if (static_cast<std::size_t>(keep_from) > audio_start_sample_) {
        const std::size_t drop =
            static_cast<std::size_t>(keep_from) - audio_start_sample_;
        if (drop < audio_.size()) {
            audio_.erase(audio_.begin(), audio_.begin() + drop);
            audio_start_sample_ += drop;
        }
    }
    return true;
}

std::vector<float> OnnxSortformerDiarizer::push_audio(
    const float* samples, std::size_t length) {
    std::vector<float> out;
    if (finished_ || samples == nullptr || length == 0) return out;
    audio_.insert(audio_.end(), samples, samples + length);
    samples_seen_ += length;
    while (advance_step(false, out)) {}
    return out;
}

std::vector<float> OnnxSortformerDiarizer::end_stream() {
    std::vector<float> out;
    if (finished_) return out;
    finished_ = true;
    while (advance_step(true, out)) {}
    return out;
}

}  // namespace speech_core
