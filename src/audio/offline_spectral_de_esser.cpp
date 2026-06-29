#include "speech_core/audio/offline_spectral_de_esser.h"

#include "ooura_fft_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace speech_core::audio {
namespace {

using DeEsser = OfflineSpectralDeEsser;
using BandConfig = DeEsser::BandConfig;
using BandDiagnostic = DeEsser::BandDiagnostic;
using FrameBandDiagnostic = DeEsser::FrameBandDiagnostic;
using Parameters = DeEsser::Parameters;
using ProcessingReport = DeEsser::ProcessingReport;
using ProcessResult = DeEsser::ProcessResult;

struct DeEsserImpl {
    static constexpr double kPi = 3.141592653589793238462643383279502884;
    static constexpr double kEpsPower = 1.0e-30;
    static constexpr double kSilenceDb = -300.0;

    class Matrix {
    public:
        Matrix() = default;

        Matrix(std::size_t rows, std::size_t cols, double initial = 0.0)
            : rows_(rows), cols_(cols), data_(rows * cols, initial) {
        }

        void resize(std::size_t rows, std::size_t cols, double initial = 0.0) {
            rows_ = rows;
            cols_ = cols;
            data_.assign(rows * cols, initial);
        }

        std::size_t rows() const noexcept { return rows_; }
        std::size_t cols() const noexcept { return cols_; }

        double& operator()(std::size_t r, std::size_t c) noexcept {
            return data_[r * cols_ + c];
        }

        double operator()(std::size_t r, std::size_t c) const noexcept {
            return data_[r * cols_ + c];
        }

        void fill(double value) {
            std::fill(data_.begin(), data_.end(), value);
        }

    private:
        std::size_t rows_ = 0;
        std::size_t cols_ = 0;
        std::vector<double> data_;
    };

    struct STFTChannel {
        std::size_t frames = 0;
        std::size_t fft_size = 0;
        std::size_t hop_size = 0;
        std::size_t padded_length = 0;
        std::size_t original_length = 0;
        std::vector<double> packed_frames;

        double* frame_data(std::size_t frame) noexcept {
            return packed_frames.data() + frame * fft_size;
        }

        const double* frame_data(std::size_t frame) const noexcept {
            return packed_frames.data() + frame * fft_size;
        }
    };

    struct LinkedDetector {
        std::size_t frames = 0;
        std::size_t fft_size = 0;
        std::size_t bins = 0;
        Matrix power;
        Matrix mag_db;
    };

    static std::vector<std::vector<double>> process_internal(
        const std::vector<std::vector<double>>& audio,
        double sample_rate,
        Parameters params,
        ProcessingReport* report) {
        if (report != nullptr)
            *report = ProcessingReport {};

        validate_input(audio, sample_rate, params);

        const std::size_t num_channels = audio.size();
        const std::size_t num_samples = audio[0].size();

        if (report != nullptr) {
            report->sample_rate = sample_rate;
            report->channels = num_channels;
            report->samples = num_samples;
        }

        if (num_samples == 0)
            return audio;

        sanitize_parameters(params, sample_rate);

        if (report != nullptr) {
            report->fft_size = params.fft_size;
            report->hop_size = params.hop_size;
            report->amount = params.amount;
        }

        const std::vector<double> window = make_sqrt_hann(params.fft_size);

        std::vector<STFTChannel> stft(num_channels);
        for (std::size_t ch = 0; ch < num_channels; ++ch)
            stft[ch] = forward_stft(audio[ch], params.fft_size, params.hop_size, window);

        LinkedDetector detector = make_linked_detector(stft, num_channels, params.fft_size);

        LinkedDetector short_detector;
        const LinkedDetector* short_detector_ptr = nullptr;
        if (params.enable_multi_resolution_detector) {
            const std::vector<double> short_window = make_sqrt_hann(params.multi_res_fft_size);
            std::vector<STFTChannel> short_stft(num_channels);
            for (std::size_t ch = 0; ch < num_channels; ++ch) {
                short_stft[ch] = forward_stft(
                    audio[ch],
                    params.multi_res_fft_size,
                    params.multi_res_hop_size,
                    short_window);
            }

            short_detector = make_linked_detector(short_stft, num_channels, params.multi_res_fft_size);
            short_detector_ptr = &short_detector;
        }

        Matrix gr_db = build_gain_reduction_mask(detector, short_detector_ptr, sample_rate, params, report);

        std::vector<std::vector<double>> out(num_channels);
        for (std::size_t ch = 0; ch < num_channels; ++ch) {
            out[ch] = inverse_stft(
                stft[ch],
                gr_db,
                params.amount,
                num_samples,
                params.fft_size,
                params.hop_size,
                window);
        }

        if (params.clip_output_to_input_peak)
            limit_to_original_peak(audio, out);

        return out;
    }

    static void validate_input(
        const std::vector<std::vector<double>>& audio,
        double sample_rate,
        const Parameters& params) {
        if (audio.empty())
            throw std::invalid_argument("OfflineSpectralDeEsser: audio has no channels");

        if (!(sample_rate > 0.0) || !std::isfinite(sample_rate))
            throw std::invalid_argument("OfflineSpectralDeEsser: invalid sample_rate");

        const std::size_t n = audio[0].size();
        for (std::size_t ch = 1; ch < audio.size(); ++ch) {
            if (audio[ch].size() != n)
                throw std::invalid_argument("OfflineSpectralDeEsser: all channels must have the same length");
        }

        if (params.fft_size < 256 || !is_power_of_two(params.fft_size))
            throw std::invalid_argument("OfflineSpectralDeEsser: fft_size must be a power of two and >= 256");

        if (params.hop_size == 0 || params.hop_size > params.fft_size)
            throw std::invalid_argument("OfflineSpectralDeEsser: hop_size must be in [1, fft_size]");
    }

    static void sanitize_parameters(Parameters& p, double sample_rate) {
        if (p.bands.empty())
            p.bands = Parameters::balanced_bands();

        p.amount = std::max(0.0, p.amount);

        p.baseline_percentile = clamp(p.baseline_percentile, 1.0, 99.0);
        p.p_rel_full_over_db = std::max(0.01, p.p_rel_full_over_db);
        p.min_noise_probability = clamp(p.min_noise_probability, 0.0, 1.0);

        p.bin_ratio = std::max(1.0, p.bin_ratio);
        p.bin_knee_db = std::max(0.0, p.bin_knee_db);

        p.hf_guard_ratio = std::max(1.0, p.hf_guard_ratio);
        p.hf_guard_knee_db = std::max(0.0, p.hf_guard_knee_db);
        p.hf_guard_max_reduction_db = std::max(0.0, p.hf_guard_max_reduction_db);

        p.frequency_smooth_width_octaves = std::max(0.0, p.frequency_smooth_width_octaves);
        p.attack_ms = std::max(0.01, p.attack_ms);
        p.release_ms = std::max(0.01, p.release_ms);
        p.gaussian_time_sigma_ms = std::max(0.0, p.gaussian_time_sigma_ms);

        p.event_min_peak_reduction_db = std::max(0.0, p.event_min_peak_reduction_db);
        p.event_relative_floor_db = std::max(0.0, p.event_relative_floor_db);

        p.psycho_focus_hz = std::max(20.0, p.psycho_focus_hz);
        p.psycho_focus_width_octaves = std::max(0.05, p.psycho_focus_width_octaves);
        p.psycho_min_weight = clamp(p.psycho_min_weight, 0.0, 1.0);

        p.anti_lisp_protect_hi_hz = std::max(0.0, p.anti_lisp_protect_hi_hz);
        p.anti_lisp_transition_hi_hz = std::max(p.anti_lisp_protect_hi_hz + 10.0, p.anti_lisp_transition_hi_hz);
        p.anti_lisp_protected_max_reduction_db = std::max(0.0, p.anti_lisp_protected_max_reduction_db);
        p.anti_lisp_transition_max_reduction_db = std::max(
            p.anti_lisp_protected_max_reduction_db,
            p.anti_lisp_transition_max_reduction_db);

        if (!is_power_of_two(p.multi_res_fft_size) || p.multi_res_fft_size < 256)
            p.multi_res_fft_size = 512;
        if (p.multi_res_hop_size == 0 || p.multi_res_hop_size > p.multi_res_fft_size)
            p.multi_res_hop_size = p.multi_res_fft_size / 4;
        p.multi_res_lo_hz = std::max(0.0, p.multi_res_lo_hz);
        p.multi_res_hi_hz = std::max(p.multi_res_lo_hz + 10.0, p.multi_res_hi_hz);
        p.multi_res_threshold_db = std::max(0.0, p.multi_res_threshold_db);
        p.multi_res_full_db = std::max(0.01, p.multi_res_full_db);
        p.multi_res_boost = std::max(0.0, p.multi_res_boost);

        const double nyquist = 0.5 * sample_rate;
        p.ref_hi_hz = std::min(p.ref_hi_hz, nyquist * 0.95);
        p.presence_hi_hz = std::min(p.presence_hi_hz, nyquist * 0.95);
        p.multi_res_hi_hz = std::min(p.multi_res_hi_hz, nyquist * 0.95);

        if (p.ref_hi_hz <= p.ref_lo_hz) {
            p.ref_lo_hz = std::max(20.0, nyquist * 0.05);
            p.ref_hi_hz = std::max(p.ref_lo_hz + 10.0, nyquist * 0.45);
        }

        if (p.presence_hi_hz <= p.presence_lo_hz) {
            p.presence_lo_hz = p.ref_lo_hz;
            p.presence_hi_hz = p.ref_hi_hz;
        }
    }

    static bool is_power_of_two(std::size_t x) noexcept {
        return x != 0 && (x & (x - 1)) == 0;
    }

    static double clamp(double x, double lo, double hi) noexcept {
        return std::max(lo, std::min(hi, x));
    }

    static double clamp01(double x) noexcept {
        return clamp(x, 0.0, 1.0);
    }

    static double smoothstep(double edge0, double edge1, double x) noexcept {
        if (edge1 <= edge0)
            return x >= edge1 ? 1.0 : 0.0;

        const double t = clamp01((x - edge0) / (edge1 - edge0));
        return t * t * (3.0 - 2.0 * t);
    }

    static double db_to_linear(double db) noexcept {
        return std::pow(10.0, db / 20.0);
    }

    static double power_to_db(double power) noexcept {
        return 10.0 * std::log10(std::max(power, kEpsPower));
    }

    static double soft_knee_over_db(double over_db, double knee_db) noexcept {
        if (knee_db <= 0.0)
            return std::max(0.0, over_db);

        const double half_knee = 0.5 * knee_db;

        if (over_db <= -half_knee)
            return 0.0;

        if (over_db >= half_knee)
            return over_db;

        const double x = over_db + half_knee;
        return (x * x) / (2.0 * knee_db);
    }

    static double compress_over_db(
        double over_db,
        double ratio,
        double knee_db,
        double max_reduction_db) noexcept {
        if (ratio <= 1.0 || max_reduction_db <= 0.0)
            return 0.0;

        const double knee_over = soft_knee_over_db(over_db, knee_db);
        const double gr_db = knee_over * (1.0 - 1.0 / ratio);

        return clamp(gr_db, 0.0, max_reduction_db);
    }

    static std::vector<double> make_sqrt_hann(std::size_t size) {
        std::vector<double> w(size);
        if (size == 0)
            return w;

        for (std::size_t n = 0; n < size; ++n) {
            const double phase = 2.0 * kPi * static_cast<double>(n) / static_cast<double>(size);
            const double hann = 0.5 - 0.5 * std::cos(phase);
            w[n] = std::sqrt(std::max(0.0, hann));
        }

        return w;
    }

    static void validate_packed_fft_buffer(const std::vector<double>& data) {
        if (data.size() < 2 || !is_power_of_two(data.size()))
            throw std::invalid_argument(
                "OfflineSpectralDeEsser: packed FFT buffer must be a power-of-two size");
    }

    class PackedFftWorkspace {
    public:
        explicit PackedFftWorkspace(std::size_t fft_size)
            : fft_size_(fft_size),
              nyquist_bin_(fft_size / 2),
              bins_(nyquist_bin_ + 1),
              input_(fft_size, 0.0f),
              real_(bins_, 0.0f),
              imag_(bins_, 0.0f),
              output_(fft_size, 0.0f) {
            engine_.init(fft_size_);
        }

        void forward(std::vector<double>& data) {
            for (std::size_t i = 0; i < fft_size_; ++i)
                input_[i] = static_cast<float>(data[i]);

            engine_.fft(input_.data(), real_.data(), imag_.data());

            data[0] = static_cast<double>(real_[0]);
            data[1] = static_cast<double>(real_[nyquist_bin_]);
            for (std::size_t k = 1; k < nyquist_bin_; ++k) {
                data[2 * k] = static_cast<double>(real_[k]);
                data[2 * k + 1] = static_cast<double>(imag_[k]);
            }
        }

        void inverse(std::vector<double>& data) {
            std::fill(real_.begin(), real_.end(), 0.0f);
            std::fill(imag_.begin(), imag_.end(), 0.0f);

            real_[0] = static_cast<float>(data[0]);
            real_[nyquist_bin_] = static_cast<float>(data[1]);
            for (std::size_t k = 1; k < nyquist_bin_; ++k) {
                real_[k] = static_cast<float>(data[2 * k]);
                imag_[k] = static_cast<float>(data[2 * k + 1]);
            }

            engine_.ifft(output_.data(), real_.data(), imag_.data());

            for (std::size_t i = 0; i < fft_size_; ++i)
                data[i] = static_cast<double>(output_[i]);
        }

    private:
        std::size_t fft_size_ = 0;
        std::size_t nyquist_bin_ = 0;
        std::size_t bins_ = 0;
        OouraFftEngine engine_;
        std::vector<float> input_;
        std::vector<float> real_;
        std::vector<float> imag_;
        std::vector<float> output_;
    };

    static void transform_packed(std::vector<double>& data, PackedFftWorkspace& fft) {
        validate_packed_fft_buffer(data);

        fft.forward(data);
    }

    static void inverse_transform_packed(std::vector<double>& data, PackedFftWorkspace& fft) {
        validate_packed_fft_buffer(data);

        fft.inverse(data);
    }

    static STFTChannel forward_stft(
        const std::vector<double>& x,
        std::size_t fft_size,
        std::size_t hop_size,
        const std::vector<double>& window) {
        const std::size_t pad = fft_size / 2;
        const std::size_t padded_length = x.size() + 2 * pad;
        const std::size_t frames = 1 + ceil_div(padded_length > fft_size ? padded_length - fft_size : 0, hop_size);
        const std::size_t total_padded_length = (frames - 1) * hop_size + fft_size;

        STFTChannel stft;
        stft.frames = frames;
        stft.fft_size = fft_size;
        stft.hop_size = hop_size;
        stft.padded_length = total_padded_length;
        stft.original_length = x.size();
        stft.packed_frames.assign(frames * fft_size, 0.0);

        PackedFftWorkspace fft(fft_size);
        std::vector<double> frame(fft_size, 0.0);

        for (std::size_t t = 0; t < frames; ++t) {
            const std::int64_t frame_start = static_cast<std::int64_t>(t * hop_size);
            std::fill(frame.begin(), frame.end(), 0.0);

            for (std::size_t n = 0; n < fft_size; ++n) {
                const std::int64_t source_index = frame_start + static_cast<std::int64_t>(n) - static_cast<std::int64_t>(pad);

                double sample = 0.0;
                if (source_index >= 0 && static_cast<std::size_t>(source_index) < x.size())
                    sample = x[static_cast<std::size_t>(source_index)];

                frame[n] = sample * window[n];
            }

            transform_packed(frame, fft);

            std::copy(frame.begin(), frame.end(), stft.frame_data(t));
        }

        return stft;
    }

    static std::size_t ceil_div(std::size_t a, std::size_t b) noexcept {
        return b == 0 ? 0 : (a + b - 1) / b;
    }

    static std::vector<double> make_frequencies(double sample_rate, std::size_t fft_size) {
        const std::size_t bins = fft_size / 2 + 1;
        std::vector<double> freq(bins);

        for (std::size_t k = 0; k < bins; ++k)
            freq[k] = sample_rate * static_cast<double>(k) / static_cast<double>(fft_size);

        return freq;
    }

    static double bin_power_from_packed(const double* packed, std::size_t fft_size, std::size_t bin) noexcept {
        const std::size_t nyquist_bin = fft_size / 2;

        if (bin == 0)
            return packed[0] * packed[0];

        if (bin == nyquist_bin)
            return packed[1] * packed[1];

        const double re = packed[2 * bin];
        const double im = packed[2 * bin + 1];
        return re * re + im * im;
    }

    static void multiply_packed_bin(
        double* packed,
        std::size_t fft_size,
        std::size_t bin,
        double gain) noexcept {
        const std::size_t nyquist_bin = fft_size / 2;

        if (bin == 0) {
            packed[0] *= gain;
            return;
        }

        if (bin == nyquist_bin) {
            packed[1] *= gain;
            return;
        }

        packed[2 * bin] *= gain;
        packed[2 * bin + 1] *= gain;
    }

    static LinkedDetector make_linked_detector(
        const std::vector<STFTChannel>& stft,
        std::size_t num_channels,
        std::size_t fft_size) {
        const std::size_t frames = stft[0].frames;
        const std::size_t bins = fft_size / 2 + 1;

        LinkedDetector det;
        det.frames = frames;
        det.fft_size = fft_size;
        det.bins = bins;
        det.power.resize(frames, bins, 0.0);
        det.mag_db.resize(frames, bins, kSilenceDb);

        for (std::size_t t = 0; t < frames; ++t) {
            for (std::size_t k = 0; k < bins; ++k) {
                double sum_power = 0.0;
                double max_power = 0.0;

                for (std::size_t ch = 0; ch < num_channels; ++ch) {
                    const double* packed = stft[ch].frame_data(t);
                    const double p = bin_power_from_packed(packed, fft_size, k);

                    sum_power += p;
                    max_power = std::max(max_power, p);
                }

                const double mean_power = sum_power / static_cast<double>(num_channels);
                const double linked_power = num_channels == 1
                    ? mean_power
                    : 0.7 * max_power + 0.3 * mean_power;

                det.power(t, k) = linked_power;
                det.mag_db(t, k) = power_to_db(linked_power);
            }
        }

        return det;
    }

    static std::vector<BandConfig> active_bands(
        const std::vector<BandConfig>& input_bands,
        double sample_rate) {
        const double usable_max_hz = 0.5 * sample_rate * 0.98;
        std::vector<BandConfig> out;
        out.reserve(input_bands.size());

        for (BandConfig b : input_bands) {
            b.lo_hz = std::max(0.0, b.lo_hz);
            b.hi_hz = std::min(b.hi_hz, usable_max_hz);

            if (b.hi_hz > b.lo_hz + 10.0 && b.max_reduction_db > 0.0)
                out.push_back(std::move(b));
        }

        return out;
    }

    struct ReductionStats {
        double max_db = 0.0;
        double mean_db = 0.0;
        double peak_hz = 0.0;
        std::size_t count = 0;
    };

    static ReductionStats reduction_stats_in_band(
        const Matrix& gr_db,
        std::size_t frame,
        const std::vector<double>& freq,
        double lo_hz,
        double hi_hz) {
        ReductionStats stats;
        double sum = 0.0;

        for (std::size_t k = 0; k < freq.size(); ++k) {
            if (freq[k] < lo_hz || freq[k] > hi_hz)
                continue;

            const double v = gr_db(frame, k);
            sum += v;
            ++stats.count;

            if (v > stats.max_db) {
                stats.max_db = v;
                stats.peak_hz = freq[k];
            }
        }

        stats.mean_db = stats.count > 0 ? sum / static_cast<double>(stats.count) : 0.0;
        return stats;
    }

    static double peak_frequency_in_band(
        const Matrix& mag_db,
        std::size_t frame,
        const std::vector<double>& freq,
        double lo_hz,
        double hi_hz) {
        double peak_db = kSilenceDb;
        double peak_hz = 0.0;

        for (std::size_t k = 0; k < freq.size(); ++k) {
            if (freq[k] < lo_hz || freq[k] > hi_hz)
                continue;

            if (mag_db(frame, k) > peak_db) {
                peak_db = mag_db(frame, k);
                peak_hz = freq[k];
            }
        }

        return peak_hz;
    }

    static double psychoacoustic_weight_for_frequency(double frequency, const Parameters& params) {
        if (!params.enable_psychoacoustic_weighting || frequency <= 0.0)
            return 1.0;

        const double octave_distance = std::log2(frequency / params.psycho_focus_hz);
        const double focus = std::exp(
            -(octave_distance * octave_distance) /
            (2.0 * params.psycho_focus_width_octaves * params.psycho_focus_width_octaves));

        return params.psycho_min_weight + (1.0 - params.psycho_min_weight) * focus;
    }

    static double anti_lisp_limit_for_frequency(double frequency, const Parameters& params) {
        if (!params.enable_anti_lisp_constraint)
            return std::numeric_limits<double>::infinity();

        if (frequency <= params.anti_lisp_protect_hi_hz)
            return params.anti_lisp_protected_max_reduction_db;

        if (frequency >= params.anti_lisp_transition_hi_hz)
            return std::numeric_limits<double>::infinity();

        const double t = smoothstep(
            params.anti_lisp_protect_hi_hz,
            params.anti_lisp_transition_hi_hz,
            frequency);

        return params.anti_lisp_protected_max_reduction_db * (1.0 - t) +
            params.anti_lisp_transition_max_reduction_db * t;
    }

    static void apply_anti_lisp_constraint(
        Matrix& gr_db,
        const std::vector<double>& freq,
        const Parameters& params) {
        if (!params.enable_anti_lisp_constraint)
            return;

        const std::size_t T = gr_db.rows();
        const std::size_t K = gr_db.cols();
        const double amount = std::max(params.amount, 1.0e-9);

        for (std::size_t k = 0; k < K; ++k) {
            const double final_limit_db = anti_lisp_limit_for_frequency(freq[k], params);
            if (!std::isfinite(final_limit_db))
                continue;

            const double mask_limit_db = final_limit_db / amount;
            for (std::size_t t = 0; t < T; ++t)
                gr_db(t, k) = std::min(gr_db(t, k), mask_limit_db);
        }
    }

    static void apply_psychoacoustic_weighting(
        Matrix& gr_db,
        const std::vector<double>& freq,
        const Parameters& params) {
        if (!params.enable_psychoacoustic_weighting)
            return;

        const std::size_t T = gr_db.rows();
        const std::size_t K = gr_db.cols();

        for (std::size_t k = 0; k < K; ++k) {
            const double weight = psychoacoustic_weight_for_frequency(freq[k], params);
            for (std::size_t t = 0; t < T; ++t)
                gr_db(t, k) *= weight;
        }
    }

    static std::vector<double> build_multi_resolution_boost(
        const LinkedDetector* short_det,
        double sample_rate,
        const Parameters& params,
        std::size_t target_frames) {
        std::vector<double> boost(target_frames, 0.0);
        if (short_det == nullptr || !params.enable_multi_resolution_detector || short_det->frames == 0)
            return boost;

        const std::vector<double> short_freq = make_frequencies(sample_rate, short_det->fft_size);
        std::vector<double> scores(short_det->frames, 0.0);
        scores.reserve(short_det->frames);

        for (std::size_t t = 0; t < short_det->frames; ++t) {
            const double target_db = rms_band_db(
                short_det->power,
                t,
                short_freq,
                params.multi_res_lo_hz,
                params.multi_res_hi_hz);

            const double ref_db = rms_band_db(
                short_det->power,
                t,
                short_freq,
                params.ref_lo_hz,
                params.ref_hi_hz);

            scores[t] = target_db - ref_db;
        }

        const double threshold_db =
            percentile(scores, 80.0, 0.0) + params.multi_res_threshold_db;
        const double full_db = threshold_db + params.multi_res_full_db;

        for (std::size_t t = 0; t < target_frames; ++t) {
            const double time_samples = static_cast<double>(t * params.hop_size);
            const auto short_frame = static_cast<std::size_t>(std::min<double>(
                static_cast<double>(short_det->frames - 1),
                std::round(time_samples / static_cast<double>(params.multi_res_hop_size))));

            boost[t] = smoothstep(threshold_db, full_db, scores[short_frame]);
        }

        return boost;
    }

    static void scale_band_frame(
        Matrix& gr_db,
        std::size_t frame,
        const std::vector<double>& shape,
        double scale) {
        const std::size_t K = gr_db.cols();
        scale = clamp01(scale);

        for (std::size_t k = 0; k < K; ++k) {
            if (shape[k] <= 0.0)
                continue;

            const double band_scale = 1.0 - shape[k] * (1.0 - scale);
            gr_db(frame, k) *= band_scale;
        }
    }

    static void refine_reduction_events(
        Matrix& gr_db,
        const std::vector<double>& freq,
        const std::vector<BandConfig>& bands,
        const std::vector<std::vector<double>>& band_shape,
        const Parameters& params) {
        if (!params.enable_event_refinement)
            return;

        const std::size_t T = gr_db.rows();
        const std::size_t B = bands.size();

        for (std::size_t b = 0; b < B; ++b) {
            std::vector<double> band_max(T, 0.0);
            for (std::size_t t = 0; t < T; ++t) {
                band_max[t] = reduction_stats_in_band(
                    gr_db,
                    t,
                    freq,
                    bands[b].lo_hz,
                    bands[b].hi_hz).max_db;
            }

            std::size_t t = 0;
            while (t < T) {
                while (t < T && band_max[t] <= 1.0e-6)
                    ++t;
                if (t >= T)
                    break;

                const std::size_t start = t;
                std::size_t end = t;
                std::size_t gap = 0;
                double peak = 0.0;

                while (t < T) {
                    peak = std::max(peak, band_max[t]);
                    if (band_max[t] > 1.0e-6) {
                        end = t;
                        gap = 0;
                    }
                    else if (++gap > params.event_max_gap_frames) {
                        break;
                    }

                    ++t;
                }

                if (peak < params.event_min_peak_reduction_db) {
                    for (std::size_t frame = start; frame <= end; ++frame)
                        scale_band_frame(gr_db, frame, band_shape[b], 0.0);
                    continue;
                }

                const double floor_db = std::max(
                    params.event_min_peak_reduction_db,
                    peak - params.event_relative_floor_db);

                for (std::size_t frame = start; frame <= end; ++frame) {
                    const double scale = smoothstep(floor_db, peak, band_max[frame]);
                    scale_band_frame(gr_db, frame, band_shape[b], scale);
                }
            }
        }
    }

    static Matrix build_gain_reduction_mask(
        const LinkedDetector& det,
        const LinkedDetector* short_det,
        double sample_rate,
        const Parameters& params,
        ProcessingReport* report) {
        const std::size_t T = det.frames;
        const std::size_t K = det.bins;
        const std::vector<double> freq = make_frequencies(sample_rate, det.fft_size);
        const std::vector<BandConfig> bands = active_bands(params.bands, sample_rate);
        const std::size_t B = bands.size();

        Matrix gr_db(T, K, 0.0);
        if (report != nullptr) {
            report->sample_rate = sample_rate;
            report->fft_size = det.fft_size;
            report->hop_size = params.hop_size;
            report->frames = T;
            report->bins = K;
            report->amount = params.amount;
            report->bands.clear();
            report->events.clear();
        }

        if (B == 0 || T == 0 || K == 0)
            return gr_db;

        std::vector<std::vector<double>> band_shape(B, std::vector<double>(K, 0.0));
        for (std::size_t b = 0; b < B; ++b) {
            const double transition = transition_hz_for_band(bands[b]);
            for (std::size_t k = 0; k < K; ++k)
                band_shape[b][k] = smooth_band_shape(freq[k], bands[b].lo_hz, bands[b].hi_hz, transition);
        }

        std::vector<double> guard_shape(K, 0.0);
        const double usable_max_hz = 0.5 * sample_rate * 0.98;
        const double guard_lo = std::max(0.0, params.hf_guard_lo_hz);
        const double guard_hi = std::min(params.hf_guard_hi_hz, usable_max_hz);
        const bool guard_active = params.enable_hf_guard && guard_hi > guard_lo + 10.0;

        if (guard_active) {
            for (std::size_t k = 0; k < K; ++k)
                guard_shape[k] = smooth_band_shape(freq[k], guard_lo, guard_hi, 1000.0);
        }

        const std::vector<double> multi_res_boost =
            build_multi_resolution_boost(short_det, sample_rate, params, T);

        std::vector<double> ref_db(T, kSilenceDb);
        std::vector<double> presence_db(T, kSilenceDb);
        Matrix band_db(T, B, kSilenceDb);
        Matrix score_db(T, B, kSilenceDb);
        Matrix flatness_db(T, B, -100.0);

        for (std::size_t t = 0; t < T; ++t) {
            ref_db[t] = rms_band_db(det.power, t, freq, params.ref_lo_hz, params.ref_hi_hz);
            presence_db[t] = rms_band_db(det.power, t, freq, params.presence_lo_hz, params.presence_hi_hz);

            for (std::size_t b = 0; b < B; ++b) {
                band_db(t, b) = rms_band_db(det.power, t, freq, bands[b].lo_hz, bands[b].hi_hz);
                score_db(t, b) = band_db(t, b) - ref_db[t];
                flatness_db(t, b) = spectral_flatness_db(det.power, t, freq, bands[b].lo_hz, bands[b].hi_hz);
            }
        }

        const double ref_noise_db = percentile(ref_db, 10.0, kSilenceDb);
        const double ref_peak_db = percentile(ref_db, 95.0, kSilenceDb);
        const double noise_relative_gate_db = ref_noise_db + params.voice_gate_above_noise_db;
        const double peak_relative_gate_db = ref_peak_db - params.voice_gate_below_peak_db;
        const double max_useful_gate_db = ref_peak_db - 6.0;
        const double voice_gate_db = std::min(
            std::max(noise_relative_gate_db, peak_relative_gate_db),
            max_useful_gate_db);

        if (report != nullptr) {
            report->ref_noise_db = ref_noise_db;
            report->ref_peak_db = ref_peak_db;
            report->voice_gate_db = voice_gate_db;
        }

        std::vector<std::uint8_t> voice_active(T, 0);
        std::size_t active_count = 0;
        for (std::size_t t = 0; t < T; ++t) {
            voice_active[t] = ref_db[t] > voice_gate_db ? 1 : 0;
            active_count += voice_active[t] ? 1 : 0;
        }

        if (report != nullptr)
            report->voice_active_frames = active_count;

        if (active_count < std::max<std::size_t>(3, T / 100)) {
            // Very short/quiet file fallback: avoid random processing.
            return gr_db;
        }

        std::vector<double> threshold_db(B, 0.0);
        std::vector<double> band_noise_db(B, kSilenceDb);

        for (std::size_t b = 0; b < B; ++b) {
            std::vector<double> scores;
            scores.reserve(active_count);

            std::vector<double> band_values;
            band_values.reserve(T);

            for (std::size_t t = 0; t < T; ++t) {
                band_values.push_back(band_db(t, b));
                if (voice_active[t])
                    scores.push_back(score_db(t, b));
            }

            const double med = percentile(scores, 50.0, -30.0);
            const double p80 = percentile(scores, 80.0, med);

            threshold_db[b] = std::max(
                med + bands[b].margin_db,
                p80 + 1.0);

            band_noise_db[b] = percentile(band_values, 10.0, kSilenceDb);
        }

        if (report != nullptr) {
            report->bands.reserve(B);
            for (std::size_t b = 0; b < B; ++b) {
                report->bands.push_back({
                    bands[b].name,
                    bands[b].lo_hz,
                    bands[b].hi_hz,
                    threshold_db[b],
                    band_noise_db[b],
                    bands[b].ratio,
                    bands[b].max_reduction_db
                });
            }
        }

        std::vector<std::uint8_t> non_sibilant_voice(T, 0);
        std::size_t non_sibilant_count = 0;

        for (std::size_t t = 0; t < T; ++t) {
            if (!voice_active[t])
                continue;

            bool non_sibilant = true;
            for (std::size_t b = 0; b < B; ++b) {
                if (score_db(t, b) > threshold_db[b] - 1.0) {
                    non_sibilant = false;
                    break;
                }
            }

            non_sibilant_voice[t] = non_sibilant ? 1 : 0;
            non_sibilant_count += non_sibilant ? 1 : 0;
        }

        if (non_sibilant_count < std::max<std::size_t>(3, active_count / 20)) {
            non_sibilant_voice = voice_active;
            non_sibilant_count = active_count;
        }

        if (report != nullptr)
            report->non_sibilant_voice_frames = non_sibilant_count;

        std::vector<double> baseline_rel_db(K, 0.0);
        for (std::size_t k = 0; k < K; ++k) {
            std::vector<double> values;
            values.reserve(non_sibilant_count);

            for (std::size_t t = 0; t < T; ++t) {
                if (non_sibilant_voice[t])
                    values.push_back(det.mag_db(t, k) - ref_db[t]);
            }

            baseline_rel_db[k] = percentile(values, params.baseline_percentile, -80.0);
        }

        // Raw time-frequency reduction mask.
        for (std::size_t t = 0; t < T; ++t) {
            if (!voice_active[t])
                continue;

            const double frame_start_seconds =
                static_cast<double>(t * params.hop_size) / sample_rate;
            const double frame_end_seconds =
                static_cast<double>((t + 1) * params.hop_size) / sample_rate;

            for (std::size_t b = 0; b < B; ++b) {
                const BandConfig& band = bands[b];

                const double band_over_db = score_db(t, b) - threshold_db[b];

                const double p_rel = smoothstep(0.0, params.p_rel_full_over_db, band_over_db);
                const double p_abs = smoothstep(
                    band_noise_db[b] + params.p_abs_low_above_noise_db,
                    band_noise_db[b] + params.p_abs_high_above_noise_db,
                    band_db(t, b));

                const double p_noise = smoothstep(
                    band.flatness_gate_db - 6.0,
                    band.flatness_gate_db,
                    flatness_db(t, b));

                const double p_sibilance = p_rel * p_abs *
                    (params.min_noise_probability + (1.0 - params.min_noise_probability) * p_noise);

                double band_gr_db = compress_over_db(
                    band_over_db,
                    band.ratio,
                    band.knee_db,
                    band.max_reduction_db);

                band_gr_db *= p_sibilance;

                if (report != nullptr && band_gr_db > 1.0e-6) {
                    FrameBandDiagnostic event;
                    event.kind = "detected";
                    event.frame_index = t;
                    event.time_seconds = frame_start_seconds;
                    event.start_seconds = frame_start_seconds;
                    event.end_seconds = frame_end_seconds;
                    event.band_name = band.name;
                    event.lo_hz = band.lo_hz;
                    event.hi_hz = band.hi_hz;
                    event.peak_hz = peak_frequency_in_band(det.mag_db, t, freq, band.lo_hz, band.hi_hz);
                    event.ref_db = ref_db[t];
                    event.presence_db = presence_db[t];
                    event.band_db = band_db(t, b);
                    event.score_db = score_db(t, b);
                    event.threshold_db = threshold_db[b];
                    event.noise_floor_db = band_noise_db[b];
                    event.flatness_db = flatness_db(t, b);
                    event.relative_probability = p_rel;
                    event.absolute_probability = p_abs;
                    event.noise_probability = p_noise;
                    event.sibilance_probability = p_sibilance;
                    event.raw_reduction_db = band_gr_db;
                    report->events.push_back(std::move(event));
                }

                if (band_gr_db <= 1.0e-6)
                    continue;

                for (std::size_t k = 0; k < K; ++k) {
                    const double shape = band_shape[b][k];
                    if (shape <= 0.0)
                        continue;

                    const double expected_bin_db =
                        ref_db[t] + baseline_rel_db[k] + margin_for_frequency(freq[k]);

                    const double bin_over_db = det.mag_db(t, k) - expected_bin_db;

                    const double bin_gr_db = compress_over_db(
                        bin_over_db,
                        params.bin_ratio,
                        params.bin_knee_db,
                        band_gr_db);

                    double candidate = std::min(band_gr_db, bin_gr_db) * shape;
                    candidate *= psychoacoustic_weight_for_frequency(freq[k], params);

                    if (freq[k] >= params.multi_res_lo_hz && freq[k] <= params.multi_res_hi_hz)
                        candidate *= 1.0 + params.multi_res_boost * multi_res_boost[t];

                    if (candidate > gr_db(t, k))
                        gr_db(t, k) = candidate;
                }
            }

            if (guard_active) {
                const double air_db = rms_band_db(det.power, t, freq, guard_lo, guard_hi);

                const double expected_air_db = std::max(
                    ref_db[t] - params.hf_expected_below_ref_db,
                    presence_db[t] - params.hf_expected_below_presence_db);

                const double hf_excess_db = air_db - expected_air_db;

                const double hf_guard_gr_db = compress_over_db(
                    hf_excess_db - params.hf_guard_threshold_db,
                    params.hf_guard_ratio,
                    params.hf_guard_knee_db,
                    params.hf_guard_max_reduction_db);

                if (hf_guard_gr_db > 1.0e-6) {
                    if (report != nullptr) {
                        FrameBandDiagnostic event;
                        event.kind = "detected";
                        event.frame_index = t;
                        event.time_seconds = frame_start_seconds;
                        event.start_seconds = frame_start_seconds;
                        event.end_seconds = frame_end_seconds;
                        event.band_name = "hf_guard";
                        event.lo_hz = guard_lo;
                        event.hi_hz = guard_hi;
                        event.peak_hz = peak_frequency_in_band(det.mag_db, t, freq, guard_lo, guard_hi);
                        event.ref_db = ref_db[t];
                        event.presence_db = presence_db[t];
                        event.band_db = air_db;
                        event.score_db = hf_excess_db;
                        event.threshold_db = params.hf_guard_threshold_db;
                        event.noise_floor_db = expected_air_db;
                        event.raw_reduction_db = hf_guard_gr_db;
                        report->events.push_back(std::move(event));
                    }

                    for (std::size_t k = 0; k < K; ++k) {
                        const double shape = guard_shape[k];
                        if (shape <= 0.0)
                            continue;

                        double candidate = hf_guard_gr_db * shape;
                        candidate *= psychoacoustic_weight_for_frequency(freq[k], params);

                        if (freq[k] >= params.multi_res_lo_hz && freq[k] <= params.multi_res_hi_hz)
                            candidate *= 1.0 + params.multi_res_boost * multi_res_boost[t];

                        if (candidate > gr_db(t, k))
                            gr_db(t, k) = candidate;
                    }
                }
            }
        }

        refine_reduction_events(gr_db, freq, bands, band_shape, params);

        if (params.enable_tonal_protection)
            apply_tonal_protection(gr_db, det.mag_db, freq, params);

        if (params.frequency_smooth_width_octaves > 0.0)
            gr_db = smooth_frequency_log_hz(gr_db, freq, params.frequency_smooth_width_octaves);

        gr_db = temporal_max_filter(gr_db, params.pre_roll_frames, params.post_roll_frames);

        gr_db = attack_release_smooth(gr_db, params.attack_ms, params.release_ms, params.hop_size, sample_rate);

        if (params.gaussian_time_sigma_ms > 0.0)
            gr_db = gaussian_time_smooth(gr_db, params.gaussian_time_sigma_ms, params.hop_size, sample_rate);

        apply_safety_clamp(gr_db, freq);
        apply_anti_lisp_constraint(gr_db, freq, params);

        if (report != nullptr) {
            for (std::size_t t = 0; t < T; ++t) {
                const double frame_start_seconds =
                    static_cast<double>(t * params.hop_size) / sample_rate;
                const double frame_end_seconds =
                    static_cast<double>((t + 1) * params.hop_size) / sample_rate;

                for (std::size_t b = 0; b < B; ++b) {
                    const ReductionStats stats =
                        reduction_stats_in_band(gr_db, t, freq, bands[b].lo_hz, bands[b].hi_hz);

                    if (stats.max_db <= 1.0e-6)
                        continue;

                    FrameBandDiagnostic event;
                    event.kind = "applied";
                    event.frame_index = t;
                    event.time_seconds = frame_start_seconds;
                    event.start_seconds = frame_start_seconds;
                    event.end_seconds = frame_end_seconds;
                    event.band_name = bands[b].name;
                    event.lo_hz = bands[b].lo_hz;
                    event.hi_hz = bands[b].hi_hz;
                    event.peak_hz = stats.peak_hz;
                    event.ref_db = ref_db[t];
                    event.presence_db = presence_db[t];
                    event.band_db = band_db(t, b);
                    event.score_db = score_db(t, b);
                    event.threshold_db = threshold_db[b];
                    event.noise_floor_db = band_noise_db[b];
                    event.flatness_db = flatness_db(t, b);
                    event.applied_max_reduction_db = stats.max_db * params.amount;
                    event.applied_mean_reduction_db = stats.mean_db * params.amount;
                    report->events.push_back(std::move(event));
                }

                bool guard_covered_by_band = false;
                for (const BandConfig& band : bands) {
                    const double overlap_lo = std::max(band.lo_hz, guard_lo);
                    const double overlap_hi = std::min(band.hi_hz, guard_hi);
                    if (overlap_hi > overlap_lo + 10.0) {
                        guard_covered_by_band = true;
                        break;
                    }
                }

                if (guard_active && !guard_covered_by_band) {
                    const ReductionStats stats =
                        reduction_stats_in_band(gr_db, t, freq, guard_lo, guard_hi);

                    if (stats.max_db > 1.0e-6) {
                        const double air_db = rms_band_db(det.power, t, freq, guard_lo, guard_hi);
                        const double expected_air_db = std::max(
                            ref_db[t] - params.hf_expected_below_ref_db,
                            presence_db[t] - params.hf_expected_below_presence_db);

                        FrameBandDiagnostic event;
                        event.kind = "applied";
                        event.frame_index = t;
                        event.time_seconds = frame_start_seconds;
                        event.start_seconds = frame_start_seconds;
                        event.end_seconds = frame_end_seconds;
                        event.band_name = "hf_guard";
                        event.lo_hz = guard_lo;
                        event.hi_hz = guard_hi;
                        event.peak_hz = stats.peak_hz;
                        event.ref_db = ref_db[t];
                        event.presence_db = presence_db[t];
                        event.band_db = air_db;
                        event.score_db = air_db - expected_air_db;
                        event.threshold_db = params.hf_guard_threshold_db;
                        event.noise_floor_db = expected_air_db;
                        event.applied_max_reduction_db = stats.max_db * params.amount;
                        event.applied_mean_reduction_db = stats.mean_db * params.amount;
                        report->events.push_back(std::move(event));
                    }
                }
            }
        }

        return gr_db;
    }

    static double rms_band_db(
        const Matrix& power,
        std::size_t frame,
        const std::vector<double>& freq,
        double lo_hz,
        double hi_hz) {
        double sum = 0.0;
        std::size_t count = 0;

        for (std::size_t k = 0; k < freq.size(); ++k) {
            if (freq[k] >= lo_hz && freq[k] <= hi_hz) {
                sum += power(frame, k);
                ++count;
            }
        }

        if (count == 0)
            return kSilenceDb;

        return power_to_db(sum / static_cast<double>(count));
    }

    static double spectral_flatness_db(
        const Matrix& power,
        std::size_t frame,
        const std::vector<double>& freq,
        double lo_hz,
        double hi_hz) {
        double log_sum = 0.0;
        double lin_sum = 0.0;
        std::size_t count = 0;

        for (std::size_t k = 0; k < freq.size(); ++k) {
            if (freq[k] >= lo_hz && freq[k] <= hi_hz) {
                const double p = std::max(power(frame, k), kEpsPower);
                log_sum += std::log(p);
                lin_sum += p;
                ++count;
            }
        }

        if (count == 0 || lin_sum <= 0.0)
            return -100.0;

        const double geometric = std::exp(log_sum / static_cast<double>(count));
        const double arithmetic = lin_sum / static_cast<double>(count);
        const double flatness = geometric / std::max(arithmetic, kEpsPower);

        return 10.0 * std::log10(std::max(flatness, kEpsPower));
    }

    static double percentile(std::vector<double> values, double p, double fallback) {
        values.erase(
            std::remove_if(values.begin(), values.end(), [](double x) { return !std::isfinite(x); }),
            values.end());

        if (values.empty())
            return fallback;

        p = clamp(p, 0.0, 100.0);
        std::sort(values.begin(), values.end());

        if (values.size() == 1)
            return values[0];

        const double pos = (p / 100.0) * static_cast<double>(values.size() - 1);
        const std::size_t i0 = static_cast<std::size_t>(std::floor(pos));
        const std::size_t i1 = std::min(i0 + 1, values.size() - 1);
        const double frac = pos - static_cast<double>(i0);

        return values[i0] * (1.0 - frac) + values[i1] * frac;
    }

    static double smooth_band_shape(
        double f,
        double lo_hz,
        double hi_hz,
        double transition_hz) noexcept {
        const double fade_in = smoothstep(lo_hz - transition_hz, lo_hz, f);
        const double fade_out = 1.0 - smoothstep(hi_hz, hi_hz + transition_hz, f);
        return clamp01(fade_in * fade_out);
    }

    static double transition_hz_for_band(const BandConfig& b) noexcept {
        if (b.lo_hz < 5000.0)
            return 400.0;
        if (b.lo_hz < 8500.0)
            return 500.0;
        if (b.lo_hz < 12500.0)
            return 700.0;
        return 1000.0;
    }

    static double margin_for_frequency(double f) noexcept {
        if (f < 5000.0)
            return 6.0;
        if (f < 8500.0)
            return 4.0;
        if (f < 12500.0)
            return 3.5;
        return 3.0;
    }

    static double max_allowed_gr_for_frequency(double f) noexcept {
        if (f < 3000.0)
            return 0.0;
        if (f < 5000.0)
            return 3.0;
        if (f < 8500.0)
            return 6.0;
        if (f < 12500.0)
            return 8.0;
        if (f < 18000.0)
            return 6.0;
        return 3.0;
    }

    static void apply_tonal_protection(
        Matrix& gr_db,
        const Matrix& mag_db,
        const std::vector<double>& freq,
        const Parameters& params) {
        const std::size_t T = gr_db.rows();
        const std::size_t K = gr_db.cols();

        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t k = 0; k < K; ++k) {
                if (freq[k] < 2500.0 || gr_db(t, k) <= 1.0e-6)
                    continue;

                const double local_median_db = median_around_row(
                    mag_db,
                    t,
                    k,
                    params.tonal_median_radius_bins);

                const double tonal_excess_db = mag_db(t, k) - local_median_db;
                const double tonal_protect = smoothstep(
                    params.tonal_protect_start_db,
                    params.tonal_protect_full_db,
                    tonal_excess_db);

                const double protection_amount = freq[k] < params.tonal_protect_split_hz
                    ? params.tonal_protect_low_high_amount
                    : params.tonal_protect_high_amount;

                gr_db(t, k) *= 1.0 - protection_amount * tonal_protect;
            }
        }
    }

    static double median_around_row(
        const Matrix& m,
        std::size_t row,
        std::size_t center,
        std::size_t radius) {
        const std::size_t K = m.cols();
        const std::size_t a = center > radius ? center - radius : 0;
        const std::size_t b = std::min(K - 1, center + radius);

        std::vector<double> values;
        values.reserve(b - a + 1);

        for (std::size_t k = a; k <= b; ++k)
            values.push_back(m(row, k));

        return percentile(values, 50.0, m(row, center));
    }

    static Matrix smooth_frequency_log_hz(
        const Matrix& in,
        const std::vector<double>& freq,
        double width_octaves) {
        const std::size_t T = in.rows();
        const std::size_t K = in.cols();

        if (width_octaves <= 0.0)
            return in;

        std::vector<std::vector<std::pair<std::size_t, double>>> weights(K);
        const double max_distance = 3.0 * width_octaves;
        const double denom = 2.0 * width_octaves * width_octaves;

        for (std::size_t k = 0; k < K; ++k) {
            if (freq[k] <= 0.0) {
                weights[k].push_back({ k, 1.0 });
                continue;
            }

            for (std::size_t j = 0; j < K; ++j) {
                if (freq[j] <= 0.0)
                    continue;

                const double dist = std::abs(std::log2(freq[j] / freq[k]));
                if (dist <= max_distance) {
                    const double w = std::exp(-(dist * dist) / denom);
                    weights[k].push_back({ j, w });
                }
            }

            if (weights[k].empty())
                weights[k].push_back({ k, 1.0 });
        }

        Matrix out(T, K, 0.0);

        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t k = 0; k < K; ++k) {
                double sum = 0.0;
                double weight_sum = 0.0;

                for (const auto& jw : weights[k]) {
                    sum += jw.second * in(t, jw.first);
                    weight_sum += jw.second;
                }

                out(t, k) = weight_sum > 0.0 ? sum / weight_sum : in(t, k);
            }
        }

        return out;
    }

    // Semantics:
    // - pre_frames: reduction may start this many frames before the detected event.
    // - post_frames: reduction may remain this many frames after the detected event.
    static Matrix temporal_max_filter(
        const Matrix& in,
        std::size_t pre_frames,
        std::size_t post_frames) {
        const std::size_t T = in.rows();
        const std::size_t K = in.cols();
        Matrix out(T, K, 0.0);

        for (std::size_t t = 0; t < T; ++t) {
            const std::size_t a = t > post_frames ? t - post_frames : 0;
            const std::size_t b = std::min(T - 1, t + pre_frames);

            for (std::size_t k = 0; k < K; ++k) {
                double m = 0.0;
                for (std::size_t i = a; i <= b; ++i)
                    m = std::max(m, in(i, k));

                out(t, k) = m;
            }
        }

        return out;
    }

    static Matrix attack_release_smooth(
        const Matrix& target,
        double attack_ms,
        double release_ms,
        std::size_t hop_size,
        double sample_rate) {
        const std::size_t T = target.rows();
        const std::size_t K = target.cols();
        Matrix out(T, K, 0.0);

        const double frame_ms = 1000.0 * static_cast<double>(hop_size) / sample_rate;
        const double attack_coeff = std::exp(-frame_ms / std::max(attack_ms, 0.01));
        const double release_coeff = std::exp(-frame_ms / std::max(release_ms, 0.01));

        for (std::size_t k = 0; k < K; ++k) {
            double state = 0.0;

            for (std::size_t t = 0; t < T; ++t) {
                const double x = target(t, k);
                const double coeff = x > state ? attack_coeff : release_coeff;
                state = coeff * state + (1.0 - coeff) * x;
                out(t, k) = state;
            }
        }

        return out;
    }

    static Matrix gaussian_time_smooth(
        const Matrix& in,
        double sigma_ms,
        std::size_t hop_size,
        double sample_rate) {
        const std::size_t T = in.rows();
        const std::size_t K = in.cols();
        Matrix out(T, K, 0.0);

        const double frame_ms = 1000.0 * static_cast<double>(hop_size) / sample_rate;
        const double sigma_frames = sigma_ms / std::max(frame_ms, 1.0e-9);

        if (sigma_frames <= 0.01)
            return in;

        const int radius = static_cast<int>(std::ceil(3.0 * sigma_frames));
        const double denom = 2.0 * sigma_frames * sigma_frames;

        std::vector<double> weights(2 * static_cast<std::size_t>(radius) + 1);
        for (int i = -radius; i <= radius; ++i)
            weights[static_cast<std::size_t>(i + radius)] = std::exp(-(static_cast<double>(i * i)) / denom);

        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t k = 0; k < K; ++k) {
                double sum = 0.0;
                double weight_sum = 0.0;

                for (int d = -radius; d <= radius; ++d) {
                    const std::int64_t ti = static_cast<std::int64_t>(t) + d;
                    if (ti < 0 || ti >= static_cast<std::int64_t>(T))
                        continue;

                    const double w = weights[static_cast<std::size_t>(d + radius)];
                    sum += w * in(static_cast<std::size_t>(ti), k);
                    weight_sum += w;
                }

                out(t, k) = weight_sum > 0.0 ? sum / weight_sum : in(t, k);
            }
        }

        return out;
    }

    static void apply_safety_clamp(Matrix& gr_db, const std::vector<double>& freq) {
        const std::size_t T = gr_db.rows();
        const std::size_t K = gr_db.cols();

        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t k = 0; k < K; ++k) {
                double v = gr_db(t, k);
                if (!std::isfinite(v) || v < 0.0)
                    v = 0.0;

                gr_db(t, k) = clamp(v, 0.0, max_allowed_gr_for_frequency(freq[k]));
            }
        }
    }

    static std::vector<double> inverse_stft(
        const STFTChannel& stft,
        const Matrix& gr_db,
        double amount,
        std::size_t output_samples,
        std::size_t fft_size,
        std::size_t hop_size,
        const std::vector<double>& window) {
        const std::size_t frames = stft.frames;
        const std::size_t bins = fft_size / 2 + 1;
        const std::size_t pad = fft_size / 2;
        const std::size_t padded_out_length = (frames - 1) * hop_size + fft_size;

        std::vector<double> out_padded(padded_out_length, 0.0);
        std::vector<double> weight(padded_out_length, 0.0);
        PackedFftWorkspace fft(fft_size);
        std::vector<double> frame(fft_size, 0.0);

        for (std::size_t t = 0; t < frames; ++t) {
            const double* source = stft.frame_data(t);
            std::copy(source, source + fft_size, frame.begin());

            for (std::size_t k = 0; k < bins; ++k) {
                const double gain = db_to_linear(-amount * gr_db(t, k));
                multiply_packed_bin(frame.data(), fft_size, k, gain);
            }

            inverse_transform_packed(frame, fft);

            const std::size_t start = t * hop_size;
            for (std::size_t n = 0; n < fft_size; ++n) {
                const std::size_t pos = start + n;
                const double v = frame[n];
                const double w = window[n];

                out_padded[pos] += v * w;
                weight[pos] += w * w;
            }
        }

        std::vector<double> out(output_samples, 0.0);
        for (std::size_t i = 0; i < output_samples; ++i) {
            const std::size_t pos = i + pad;
            if (pos < out_padded.size() && weight[pos] > 1.0e-12)
                out[i] = out_padded[pos] / weight[pos];
            else
                out[i] = 0.0;
        }

        return out;
    }

    static void limit_to_original_peak(
        const std::vector<std::vector<double>>& in,
        std::vector<std::vector<double>>& out) {
        double in_peak = 0.0;
        double out_peak = 0.0;

        for (const auto& ch : in)
            for (double x : ch)
                in_peak = std::max(in_peak, std::abs(x));

        for (const auto& ch : out)
            for (double x : ch)
                out_peak = std::max(out_peak, std::abs(x));

        if (out_peak > in_peak && out_peak > 1.0e-30) {
            const double g = in_peak / out_peak;
            for (auto& ch : out)
                for (double& x : ch)
                    x *= g;
        }
    }

};

}  // namespace

OfflineSpectralDeEsser::Parameters OfflineSpectralDeEsser::cli_default_parameters() {
    Parameters params = Parameters::red_focus_tight();
    params.fft_size = 1024;
    params.hop_size = 128;
    params.amount *= 1.60;
    params.clip_output_to_input_peak = true;
    return params;
}

std::vector<std::vector<double>> OfflineSpectralDeEsser::process(
    const std::vector<std::vector<double>>& audio,
    double sample_rate,
    Parameters params) {
    return DeEsserImpl::process_internal(audio, sample_rate, std::move(params), nullptr);
}

OfflineSpectralDeEsser::ProcessResult OfflineSpectralDeEsser::process_with_report(
    const std::vector<std::vector<double>>& audio,
    double sample_rate,
    Parameters params) {
    ProcessResult result;
    result.audio = DeEsserImpl::process_internal(
        audio, sample_rate, std::move(params), &result.report);
    return result;
}

void OfflineSpectralDeEsser::process_in_place(
    std::vector<std::vector<double>>& audio,
    double sample_rate,
    Parameters params) {
    audio = process(audio, sample_rate, std::move(params));
}

std::vector<float> OfflineSpectralDeEsser::process_mono(
    const float* samples,
    std::size_t sample_count,
    double sample_rate) {
    return process_mono(samples, sample_count, sample_rate, cli_default_parameters());
}

std::vector<float> OfflineSpectralDeEsser::process_mono(
    const float* samples,
    std::size_t sample_count,
    double sample_rate,
    Parameters params) {
    if (samples == nullptr) {
        if (sample_count == 0) {
            return {};
        }

        throw std::invalid_argument("OfflineSpectralDeEsser: samples is null");
    }

    std::vector<std::vector<double>> audio(1);
    auto& channel = audio[0];
    channel.reserve(sample_count);

    for (std::size_t i = 0; i < sample_count; ++i) {
        channel.push_back(static_cast<double>(samples[i]));
    }

    const auto processed = process(audio, sample_rate, std::move(params));

    std::vector<float> out;
    out.reserve(sample_count);
    for (double sample : processed[0]) {
        out.push_back(static_cast<float>(sample));
    }

    return out;
}

}  // namespace speech_core::audio
