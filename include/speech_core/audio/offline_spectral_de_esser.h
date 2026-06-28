#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace speech_core::audio {

class OfflineSpectralDeEsser {
public:
    struct BandConfig {
        std::string name;

        double lo_hz = 0.0;
        double hi_hz = 0.0;

        double ratio = 2.0;
        double knee_db = 6.0;
        double max_reduction_db = 6.0;

        // Used in adaptive threshold:
        // threshold = max(median(score) + margin_db, p80(score) + 1 dB)
        double margin_db = 4.0;

        // Spectral flatness gate in dB. Closer to 0 dB means more noise-like.
        // Typical sibilant flatness is much higher than voiced harmonic flatness.
        double flatness_gate_db = -18.0;
    };

    struct Parameters {
        std::size_t fft_size = 1024;
        std::size_t hop_size = 128;

        // Global intensity multiplier for the final reduction mask.
        // 0.70-0.85 = very natural, 1.0 = balanced, 1.15-1.25 = repair.
        double amount = 1.0;

        // Analysis bands. They are clipped automatically to usable Nyquist.
        std::vector<BandConfig> bands;

        // Reference bands.
        double ref_lo_hz = 500.0;
        double ref_hi_hz = 4000.0;
        double presence_lo_hz = 2000.0;
        double presence_hi_hz = 5000.0;

        // Voice gate from ref band statistics.
        double voice_gate_above_noise_db = 12.0;
        double voice_gate_below_peak_db = 50.0;

        // Per-bin adaptive baseline.
        double baseline_percentile = 60.0;

        // Sibilance probability shaping.
        double p_rel_full_over_db = 6.0;
        double p_abs_low_above_noise_db = 8.0;
        double p_abs_high_above_noise_db = 20.0;
        double min_noise_probability = 0.35;

        // Per-bin compression inside an active band.
        double bin_ratio = 2.5;
        double bin_knee_db = 4.0;

        // Synthetic HF guard.
        bool enable_hf_guard = true;
        double hf_guard_lo_hz = 12500.0;
        double hf_guard_hi_hz = 18000.0;
        double hf_expected_below_ref_db = 30.0;
        double hf_expected_below_presence_db = 22.0;
        double hf_guard_threshold_db = 2.0;
        double hf_guard_ratio = 2.0;
        double hf_guard_knee_db = 8.0;
        double hf_guard_max_reduction_db = 5.0;

        // Tonal/harmonic protection.
        bool enable_tonal_protection = true;
        std::size_t tonal_median_radius_bins = 5;
        double tonal_protect_start_db = 4.0;
        double tonal_protect_full_db = 12.0;
        double tonal_protect_low_high_amount = 0.60;  // below tonal_protect_split_hz
        double tonal_protect_high_amount = 0.25;     // above tonal_protect_split_hz
        double tonal_protect_split_hz = 8500.0;

        // Mask smoothing.
        double frequency_smooth_width_octaves = 0.08;
        std::size_t pre_roll_frames = 1;     // non-causal lookahead expansion
        std::size_t post_roll_frames = 3;    // trailing expansion
        double attack_ms = 1.0;
        double release_ms = 70.0;
        double gaussian_time_sigma_ms = 4.0;

        // Event refinement: reject weak smeared tails and keep reduction tied
        // to short, local high-frequency events.
        bool enable_event_refinement = false;
        double event_min_peak_reduction_db = 0.45;
        double event_relative_floor_db = 8.0;
        std::size_t event_max_gap_frames = 1;

        // Psychoacoustic focus: bias the mask toward the audibly sharp HF
        // island region instead of treating every Hz equally.
        bool enable_psychoacoustic_weighting = false;
        double psycho_focus_hz = 9800.0;
        double psycho_focus_width_octaves = 0.55;
        double psycho_min_weight = 0.55;

        // Anti-lisp safety budget. Values are final dB after amount.
        bool enable_anti_lisp_constraint = false;
        double anti_lisp_protect_hi_hz = 7500.0;
        double anti_lisp_transition_hi_hz = 8800.0;
        double anti_lisp_protected_max_reduction_db = 1.1;
        double anti_lisp_transition_max_reduction_db = 4.0;

        // Short-window HF detector used as a second opinion for tiny islands.
        bool enable_multi_resolution_detector = false;
        std::size_t multi_res_fft_size = 512;
        std::size_t multi_res_hop_size = 64;
        double multi_res_lo_hz = 8800.0;
        double multi_res_hi_hz = 11500.0;
        double multi_res_threshold_db = 1.5;
        double multi_res_full_db = 7.0;
        double multi_res_boost = 0.18;

        // Safety.
        bool clip_output_to_input_peak = false;

        Parameters()
            : bands(balanced_bands()) {
        }

        static std::vector<BandConfig> balanced_bands() {
            return {
                { "harsh_3_5",       3000.0,  5000.0,  2.0, 5.0, 3.0, 6.0, -20.0 },
                { "sibilance_5_8k5", 5000.0,  8500.0,  4.0, 5.0, 6.0, 4.0, -18.0 },
                { "hiss_8k5_12k5",   8500.0, 12500.0,  3.5, 6.0, 7.0, 4.0, -17.0 },
                { "air_12k5_18",    12500.0, 18000.0,  2.0, 8.0, 5.0, 3.0, -16.0 }
            };
        }

        static Parameters natural() {
            Parameters p;
            p.amount = 0.80;
            p.bands = {
                { "harsh_3_5",       3000.0,  5000.0,  1.8, 6.0, 2.0, 7.0, -20.0 },
                { "sibilance_5_8k5", 5000.0,  8500.0,  3.0, 6.0, 4.0, 5.0, -18.0 },
                { "hiss_8k5_12k5",   8500.0, 12500.0,  3.0, 7.0, 5.0, 5.0, -17.0 },
                { "air_12k5_18",    12500.0, 18000.0,  1.8, 9.0, 3.0, 4.0, -16.0 }
            };
            p.hf_guard_max_reduction_db = 3.0;
            return p;
        }

        static Parameters balanced() {
            return Parameters();
        }

        static Parameters tts_repair() {
            Parameters p;
            p.amount = 1.15;
            p.bands = {
                { "harsh_3_5",       3000.0,  5000.0,  2.2, 5.0, 3.0, 5.0, -20.0 },
                { "sibilance_5_8k5", 5000.0,  8500.0,  4.5, 5.0, 7.0, 3.5, -18.0 },
                { "hiss_8k5_12k5",   8500.0, 12500.0,  4.0, 6.0, 9.0, 3.0, -17.0 },
                { "air_12k5_18",    12500.0, 18000.0,  2.2, 8.0, 6.0, 2.5, -16.0 }
            };
            p.hf_guard_max_reduction_db = 6.0;
            p.hf_guard_threshold_db = 1.0;
            return p;
        }

        static Parameters speech_safe() {
            Parameters p;
            p.amount = 1.15;
            p.bands = {
                { "harsh_3_5",          3000.0,  5000.0,  2.0, 6.0, 2.2, 6.5, -20.0 },
                { "sibilance_5_7k2",    5000.0,  7200.0,  3.2, 6.0, 3.6, 5.0, -16.0 },
                { "sibilance_7k2_8k5",  7200.0,  8500.0,  4.0, 5.5, 5.2, 3.8, -17.0 },
                { "hiss_8k5_12k5",      8500.0, 12500.0,  4.0, 6.0, 8.5, 3.0, -17.0 },
                { "air_12k5_18",       12500.0, 18000.0,  2.2, 8.0, 6.0, 2.5, -16.0 }
            };

            p.p_rel_full_over_db = 5.5;
            p.p_abs_low_above_noise_db = 7.0;
            p.p_abs_high_above_noise_db = 20.0;
            p.hf_guard_threshold_db = 1.0;
            p.hf_guard_max_reduction_db = 6.0;

            p.tonal_protect_low_high_amount = 0.72;
            p.tonal_protect_high_amount = 0.32;
            p.release_ms = 85.0;
            return p;
        }

        static Parameters anti_lisp() {
            Parameters p;
            p.amount = 1.10;
            p.bands = {
                { "harsh_3_5",          3000.0,  5000.0,  1.8, 7.0, 1.4, 8.0, -18.0 },
                { "sibilance_5_7k5",    5000.0,  7500.0,  2.4, 7.0, 2.0, 7.0, -14.0 },
                { "sibilance_7k5_8k5",  7500.0,  8500.0,  3.0, 6.0, 3.0, 5.0, -16.0 },
                { "hiss_8k5_12k5",      8500.0, 12500.0,  4.2, 6.0, 7.5, 3.0, -17.0 },
                { "air_12k5_18",       12500.0, 18000.0,  2.2, 8.0, 5.5, 2.5, -16.0 }
            };

            p.p_rel_full_over_db = 6.0;
            p.p_abs_low_above_noise_db = 8.0;
            p.p_abs_high_above_noise_db = 21.0;
            p.min_noise_probability = 0.25;

            p.hf_guard_threshold_db = 1.5;
            p.hf_guard_max_reduction_db = 5.5;

            p.tonal_protect_low_high_amount = 0.85;
            p.tonal_protect_high_amount = 0.42;
            p.pre_roll_frames = 1;
            p.post_roll_frames = 2;
            p.release_ms = 55.0;
            p.gaussian_time_sigma_ms = 3.0;
            return p;
        }

        static Parameters anti_lisp_plus() {
            Parameters p;
            p.amount = 1.10;
            p.bands = {
                { "harsh_3_5",          3000.0,  5000.0,  1.6, 7.5, 1.1, 8.5, -17.0 },
                { "sibilance_5_7k5",    5000.0,  7500.0,  2.0, 8.0, 1.3, 8.5, -12.5 },
                { "sibilance_7k5_8k5",  7500.0,  8500.0,  2.8, 6.5, 2.5, 5.5, -15.0 },
                { "hiss_8k5_12k5",      8500.0, 12500.0,  4.2, 6.0, 7.8, 2.8, -17.0 },
                { "air_12k5_18",       12500.0, 18000.0,  2.2, 8.0, 5.8, 2.5, -16.0 }
            };

            p.p_rel_full_over_db = 6.5;
            p.p_abs_low_above_noise_db = 8.5;
            p.p_abs_high_above_noise_db = 21.5;
            p.min_noise_probability = 0.18;

            p.hf_guard_threshold_db = 1.4;
            p.hf_guard_max_reduction_db = 5.8;

            p.tonal_protect_low_high_amount = 0.92;
            p.tonal_protect_high_amount = 0.45;
            p.pre_roll_frames = 1;
            p.post_roll_frames = 2;
            p.release_ms = 50.0;
            p.gaussian_time_sigma_ms = 2.5;
            return p;
        }

        static Parameters red_focus() {
            Parameters p;
            p.amount = 1.10;
            p.bands = {
                { "edge_5k8_7k5",       5800.0,  7500.0,  2.0, 8.0, 1.4, 8.5, -12.0 },
                { "sibilance_7k5_8k8",  7500.0,  8800.0,  3.2, 6.5, 3.2, 5.0, -15.0 },
                { "islands_8k8_11k5",   8800.0, 11500.0,  5.0, 5.5, 8.2, 2.2, -17.0 },
                { "islands_11k5_13k8", 11500.0, 13800.0,  4.2, 6.5, 6.5, 2.2, -17.0 }
            };

            p.ref_lo_hz = 500.0;
            p.ref_hi_hz = 4200.0;
            p.presence_lo_hz = 2200.0;
            p.presence_hi_hz = 5200.0;

            p.p_rel_full_over_db = 4.5;
            p.p_abs_low_above_noise_db = 7.0;
            p.p_abs_high_above_noise_db = 19.0;
            p.min_noise_probability = 0.30;

            p.hf_guard_lo_hz = 9000.0;
            p.hf_guard_hi_hz = 13800.0;
            p.hf_expected_below_ref_db = 27.0;
            p.hf_expected_below_presence_db = 20.0;
            p.hf_guard_threshold_db = 1.0;
            p.hf_guard_max_reduction_db = 5.8;

            p.tonal_protect_low_high_amount = 0.95;
            p.tonal_protect_high_amount = 0.35;
            p.tonal_protect_split_hz = 7800.0;

            p.frequency_smooth_width_octaves = 0.06;
            p.pre_roll_frames = 1;
            p.post_roll_frames = 2;
            p.attack_ms = 0.8;
            p.release_ms = 45.0;
            p.gaussian_time_sigma_ms = 2.5;

            p.enable_event_refinement = true;
            p.event_min_peak_reduction_db = 0.35;
            p.event_relative_floor_db = 7.0;
            p.event_max_gap_frames = 1;

            p.enable_psychoacoustic_weighting = true;
            p.psycho_focus_hz = 9800.0;
            p.psycho_focus_width_octaves = 0.50;
            p.psycho_min_weight = 0.55;

            p.enable_anti_lisp_constraint = true;
            p.anti_lisp_protect_hi_hz = 7500.0;
            p.anti_lisp_transition_hi_hz = 8800.0;
            p.anti_lisp_protected_max_reduction_db = 1.1;
            p.anti_lisp_transition_max_reduction_db = 4.1;

            p.enable_multi_resolution_detector = true;
            p.multi_res_fft_size = 512;
            p.multi_res_hop_size = 64;
            p.multi_res_lo_hz = 8800.0;
            p.multi_res_hi_hz = 11500.0;
            p.multi_res_threshold_db = 1.0;
            p.multi_res_full_db = 6.5;
            p.multi_res_boost = 0.16;
            return p;
        }

        static Parameters red_focus_legacy() {
            Parameters p;
            p.amount = 1.10;
            p.bands = {
                { "edge_5k8_7k5",       5800.0,  7500.0,  2.0, 8.0, 1.4, 8.5, -12.0 },
                { "sibilance_7k5_8k8",  7500.0,  8800.0,  3.2, 6.5, 3.2, 5.0, -15.0 },
                { "islands_8k8_11k5",   8800.0, 11500.0,  5.0, 5.5, 8.2, 2.2, -17.0 },
                { "islands_11k5_13k8", 11500.0, 13800.0,  4.2, 6.5, 6.5, 2.2, -17.0 }
            };

            p.ref_lo_hz = 500.0;
            p.ref_hi_hz = 4200.0;
            p.presence_lo_hz = 2200.0;
            p.presence_hi_hz = 5200.0;

            p.p_rel_full_over_db = 4.5;
            p.p_abs_low_above_noise_db = 7.0;
            p.p_abs_high_above_noise_db = 19.0;
            p.min_noise_probability = 0.30;

            p.hf_guard_lo_hz = 9000.0;
            p.hf_guard_hi_hz = 13800.0;
            p.hf_expected_below_ref_db = 27.0;
            p.hf_expected_below_presence_db = 20.0;
            p.hf_guard_threshold_db = 1.0;
            p.hf_guard_max_reduction_db = 5.8;

            p.tonal_protect_low_high_amount = 0.95;
            p.tonal_protect_high_amount = 0.35;
            p.tonal_protect_split_hz = 7800.0;

            p.frequency_smooth_width_octaves = 0.06;
            p.pre_roll_frames = 1;
            p.post_roll_frames = 2;
            p.attack_ms = 0.8;
            p.release_ms = 45.0;
            p.gaussian_time_sigma_ms = 2.5;
            return p;
        }

        static Parameters red_focus_tight() {
            Parameters p = red_focus();
            p.anti_lisp_protected_max_reduction_db = 0.75;
            p.anti_lisp_transition_max_reduction_db = 3.30;
            p.psycho_min_weight = 0.50;
            p.event_relative_floor_db = 6.0;
            p.multi_res_boost = 0.18;
            return p;
        }

        static Parameters aggressive() {
            Parameters p;
            p.amount = 1.35;
            p.bands = {
                { "harsh_3_5",          3000.0,  5000.0,  2.8, 4.0,  5.0, 2.5, -22.0 },
                { "sibilance_4k2_8k5", 4200.0,  8500.0,  6.0, 4.0, 11.0, 1.5, -22.0 },
                { "hiss_8k5_12k5",     8500.0, 12500.0,  5.5, 5.0, 12.0, 1.5, -20.0 },
                { "air_12k5_18",      12500.0, 18000.0,  3.0, 7.0,  8.0, 1.0, -18.0 }
            };

            p.p_rel_full_over_db = 3.5;
            p.p_abs_low_above_noise_db = 4.0;
            p.p_abs_high_above_noise_db = 14.0;
            p.min_noise_probability = 0.55;

            p.hf_guard_threshold_db = 0.0;
            p.hf_guard_max_reduction_db = 8.0;

            p.tonal_protect_low_high_amount = 0.35;
            p.tonal_protect_high_amount = 0.15;

            p.pre_roll_frames = 2;
            p.post_roll_frames = 5;
            p.release_ms = 90.0;
            p.gaussian_time_sigma_ms = 5.0;
            return p;
        }
    };

    struct BandDiagnostic {
        std::string name;
        double lo_hz = 0.0;
        double hi_hz = 0.0;
        double threshold_db = 0.0;
        double noise_floor_db = -300.0;
        double ratio = 1.0;
        double max_reduction_db = 0.0;
    };

    struct FrameBandDiagnostic {
        // "detected" is the raw detector decision before mask smoothing.
        // "applied" is the final reduction mask after protection/smoothing/clamp.
        std::string kind;

        std::size_t frame_index = 0;
        double time_seconds = 0.0;
        double start_seconds = 0.0;
        double end_seconds = 0.0;

        std::string band_name;
        double lo_hz = 0.0;
        double hi_hz = 0.0;
        double peak_hz = 0.0;

        double ref_db = -300.0;
        double presence_db = -300.0;
        double band_db = -300.0;
        double score_db = 0.0;
        double threshold_db = 0.0;
        double noise_floor_db = -300.0;
        double flatness_db = -100.0;

        double relative_probability = 0.0;
        double absolute_probability = 0.0;
        double noise_probability = 0.0;
        double sibilance_probability = 0.0;

        double raw_reduction_db = 0.0;
        double applied_max_reduction_db = 0.0;
        double applied_mean_reduction_db = 0.0;
    };

    struct ProcessingReport {
        double sample_rate = 0.0;
        std::size_t channels = 0;
        std::size_t samples = 0;
        std::size_t fft_size = 0;
        std::size_t hop_size = 0;
        std::size_t frames = 0;
        std::size_t bins = 0;
        double amount = 0.0;

        double ref_noise_db = -300.0;
        double ref_peak_db = -300.0;
        double voice_gate_db = -300.0;
        std::size_t voice_active_frames = 0;
        std::size_t non_sibilant_voice_frames = 0;

        std::vector<BandDiagnostic> bands;
        std::vector<FrameBandDiagnostic> events;
    };

    struct ProcessResult {
        std::vector<std::vector<double>> audio;
        ProcessingReport report;
    };

    static Parameters cli_default_parameters();

    // Input layout: audio[channel][sample], non-interleaved.
    // Output layout is the same. All channels must have the same sample count.
    static std::vector<std::vector<double>> process(
        const std::vector<std::vector<double>>& audio,
        double sample_rate,
        Parameters params = Parameters());

    static ProcessResult process_with_report(
        const std::vector<std::vector<double>>& audio,
        double sample_rate,
        Parameters params = Parameters());

    static void process_in_place(
        std::vector<std::vector<double>>& audio,
        double sample_rate,
        Parameters params = Parameters());

    static std::vector<float> process_mono(
        const float* samples,
        std::size_t sample_count,
        double sample_rate);

    static std::vector<float> process_mono(
        const float* samples,
        std::size_t sample_count,
        double sample_rate,
        Parameters params);
};

}  // namespace speech_core::audio
