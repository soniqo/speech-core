#include "tflite_c_api_minimal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double proc_memory_mib(const std::string& field) {
#if defined(__linux__) || defined(__ANDROID__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind(field + ":", 0) != 0) continue;
        std::istringstream values(line.substr(field.size() + 1));
        double kib = 0.0;
        values >> kib;
        return kib / 1024.0;
    }
#else
    (void)field;
#endif
    return 0.0;
}
double percentile(const std::vector<double>& sorted, double fraction) {
    const size_t rank = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(fraction * sorted.size())));
    return sorted[std::min(rank - 1, sorted.size() - 1)];
}

void check(TfLiteStatus status, const char* operation) {
    if (status != kTfLiteOk) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            throw std::invalid_argument(
                "usage: speech_litert_tensor_bench MODEL [THREADS] [WARMUP] [RUNS]");
        }
        const std::string model_path = argv[1];
        const int threads = argc > 2 ? std::stoi(argv[2]) : 8;
        const int warmup = argc > 3 ? std::stoi(argv[3]) : 1;
        const int runs = argc > 4 ? std::stoi(argv[4]) : 10;
        if (threads <= 0 || warmup < 0 || runs <= 0) {
            throw std::invalid_argument("invalid threads/warmup/runs");
        }

        const auto load_start = std::chrono::steady_clock::now();
        TfLiteModel* model = TfLiteModelCreateFromFile(model_path.c_str());
        if (!model) throw std::runtime_error("failed to load model");
        TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(options, threads);
        TfLiteXNNPackDelegateOptions xnnpack_options =
            TfLiteXNNPackDelegateOptionsDefault();
        xnnpack_options.num_threads = threads;
        TfLiteOpaqueDelegate* delegate =
            TfLiteXNNPackDelegateCreate(&xnnpack_options);
        if (!delegate) throw std::runtime_error("failed to create XNNPACK");
        TfLiteInterpreterOptionsAddDelegate(options, delegate);
        TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
        TfLiteInterpreterOptionsDelete(options);
        if (!interpreter) throw std::runtime_error("failed to create interpreter");
        check(TfLiteInterpreterAllocateTensors(interpreter), "AllocateTensors");
        const auto load_stop = std::chrono::steady_clock::now();

        if (TfLiteInterpreterGetInputTensorCount(interpreter) != 1 ||
            TfLiteInterpreterGetOutputTensorCount(interpreter) < 1) {
            throw std::runtime_error("expected one input and at least one output");
        }
        TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
        const size_t input_bytes = TfLiteTensorByteSize(input);
        if (input_bytes == 0 || input_bytes % sizeof(float) != 0) {
            throw std::runtime_error("expected non-empty Float32 input");
        }
        std::vector<float> values(input_bytes / sizeof(float));
        uint32_t state = 1234;
        for (float& value : values) {
            state = state * 1664525u + 1013904223u;
            value = static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
        }
        check(TfLiteTensorCopyFromBuffer(input, values.data(), input_bytes),
              "TensorCopyFromBuffer");

        const auto invoke = [&] {
            check(TfLiteInterpreterInvoke(interpreter), "Invoke");
        };
        const auto cold_start = std::chrono::steady_clock::now();
        invoke();
        const auto cold_stop = std::chrono::steady_clock::now();
        for (int i = 0; i < warmup; ++i) invoke();
        std::vector<double> elapsed;
        elapsed.reserve(static_cast<size_t>(runs));
        for (int i = 0; i < runs; ++i) {
            const auto start = std::chrono::steady_clock::now();
            invoke();
            const auto stop = std::chrono::steady_clock::now();
            elapsed.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());
        }
        std::sort(elapsed.begin(), elapsed.end());

        const double load_ms =
            std::chrono::duration<double, std::milli>(load_stop - load_start).count();
        const double cold_ms =
            std::chrono::duration<double, std::milli>(cold_stop - cold_start).count();
        std::cout << "model=" << model_path << " threads=" << threads
                  << " input_bytes=" << input_bytes << '\n'
                  << "load_ms=" << std::fixed << std::setprecision(3) << load_ms
                  << " cold_ms=" << cold_ms
                  << " p50_ms=" << percentile(elapsed, 0.50)
                  << " p90_ms=" << percentile(elapsed, 0.90)
                  << " current_rss_mib=" << std::setprecision(1)
                  << proc_memory_mib("VmRSS")
                  << " peak_rss_mib=" << proc_memory_mib("VmHWM") << '\n';

        TfLiteInterpreterDelete(interpreter);
        TfLiteXNNPackDelegateDelete(delegate);
        TfLiteModelDelete(model);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
