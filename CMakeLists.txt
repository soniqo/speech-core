cmake_minimum_required(VERSION 3.16)
project(speech_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(SPEECH_CORE_BUILD_TESTS "Build tests" ON)
option(SPEECH_CORE_WITH_ONNX "Build the speech_core_models target with ONNX Runtime reference implementations (Silero VAD, Parakeet STT, Kokoro TTS, DeepFilter)" OFF)
option(SPEECH_CORE_WITH_CUDA "Enable NVIDIA GPU execution providers (CUDA / TensorRT) in the ONNX engine. Requires SPEECH_CORE_WITH_ONNX=ON and ORT_DIR pointing at a GPU-enabled ONNX Runtime (onnxruntime-gpu). Selection is further gated at runtime by env SPEECH_CORE_ORT_PROVIDER and provider availability, with automatic CPU fallback." OFF)
option(SPEECH_CORE_WITH_LITERT "Build the speech_core_models_litert target with LiteRT (TFLite) reference implementations (Silero VAD, Parakeet STT)" OFF)
option(SPEECH_CORE_BUILD_EXAMPLES "Build examples/ (Linux C ABI lib, ALSA demo, CLI tools, integration tests). Requires SPEECH_CORE_WITH_ONNX=ON." OFF)

if(SPEECH_CORE_WITH_CUDA AND NOT SPEECH_CORE_WITH_ONNX)
    message(FATAL_ERROR "SPEECH_CORE_WITH_CUDA=ON requires SPEECH_CORE_WITH_ONNX=ON "
                        "(CUDA / TensorRT EPs are ONNX-Runtime-only; LiteRT cannot use CUDA).")
endif()

# ---------------------------------------------------------------------------
# speech_core — orchestration core (always built, no ML deps)
# ---------------------------------------------------------------------------

add_library(speech_core
    src/pipeline/voice_pipeline.cpp
    src/pipeline/turn_detector.cpp
    src/pipeline/speech_queue.cpp
    src/pipeline/conversation_context.cpp
    src/vad/streaming_vad.cpp
    src/audio/audio_buffer.cpp
    src/audio/resampler.cpp
    src/audio/pcm_codec.cpp
    src/audio/fft.cpp
    src/audio/mel.cpp
    src/audio/stft.cpp
    src/tools/tool_registry.cpp
    src/tools/intent_matcher.cpp
    src/tools/tool_executor.cpp
    src/diarization/diarization_pipeline.cpp
    src/speech_core_c.cpp
)

target_include_directories(speech_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(speech_core PRIVATE /W4)
    # cmd.exe / MSVC don't define M_PI etc. without this; we still want it
    # belt-and-suspenders even though sources now use local kPi constants.
    target_compile_definitions(speech_core PRIVATE _USE_MATH_DEFINES NOMINMAX)
else()
    target_compile_options(speech_core PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O2>
    )
endif()

# Build with -fPIC so the static library can be linked into shared libraries
# (the Linux examples libspeech.so, the Android JNI .so, etc.). Apple's linker
# doesn't require this; GNU ld does, and complains with "relocation R_X86_64_*
# can not be used when making a shared object; recompile with -fPIC".
# No-op on Windows (DLLs use a different model), but harmless.
set_target_properties(speech_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# speech_core_models — ONNX Runtime reference implementations (optional)
# ---------------------------------------------------------------------------

if(SPEECH_CORE_WITH_ONNX)
    if(NOT DEFINED ORT_DIR)
        message(FATAL_ERROR "SPEECH_CORE_WITH_ONNX=ON requires ORT_DIR (path to ONNX Runtime with include/ and lib/, plus onnxruntime.dll/.lib on Windows)")
    endif()
    # Imported targets' INTERFACE_INCLUDE_DIRECTORIES must be absolute.
    get_filename_component(ORT_DIR "${ORT_DIR}" ABSOLUTE)

    # ONNX Runtime — imported shared library.
    # On Windows we need IMPORTED_LOCATION (the .dll, loaded at runtime) AND
    # IMPORTED_IMPLIB (the .lib import library, used at link time). Everywhere
    # else IMPORTED_LOCATION points directly at the shared object.
    add_library(onnxruntime SHARED IMPORTED)
    if(WIN32)
        set(_ORT_LIB    "${ORT_DIR}/lib/onnxruntime.dll")
        set(_ORT_IMPLIB "${ORT_DIR}/lib/onnxruntime.lib")
    elseif(APPLE)
        set(_ORT_LIB "${ORT_DIR}/lib/libonnxruntime.dylib")
    elseif(ANDROID)
        set(_ORT_LIB "${ORT_DIR}/lib/${ANDROID_ABI}/libonnxruntime.so")
    else()
        set(_ORT_LIB "${ORT_DIR}/lib/libonnxruntime.so")
    endif()
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${_ORT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
    )
    if(WIN32)
        set_target_properties(onnxruntime PROPERTIES IMPORTED_IMPLIB "${_ORT_IMPLIB}")
    endif()

    add_library(speech_core_models
        src/models/silero/silero_vad.cpp
        src/models/parakeet/parakeet_stt.cpp
        src/models/kokoro/kokoro_tts.cpp
        src/models/kokoro/kokoro_phonemizer.cpp
        src/models/kokoro/kokoro_multilingual.cpp
        src/models/deepfilter/deepfilter.cpp
    )

    target_include_directories(speech_core_models
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )

    if(MSVC)
        target_compile_options(speech_core_models PRIVATE /W4)
        target_compile_definitions(speech_core_models PRIVATE _USE_MATH_DEFINES NOMINMAX)
    else()
        target_compile_options(speech_core_models PRIVATE
            -Wall -Wextra
            $<$<CONFIG:Release>:-O2>
        )
    endif()

    # See speech_core above — same reason.
    set_target_properties(speech_core_models PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # NVIDIA GPU execution providers (CUDA / TensorRT). Compile-time gate only:
    # ORT_DIR must point at a GPU-enabled onnxruntime so the CUDA/TensorRT EP
    # symbols are present in libonnxruntime at link/load time. The V2 provider
    # API used in onnx_engine.h returns failure (not a crash) on a CPU-only
    # runtime, so a binary built with this flag still runs on a CPU-only host;
    # runtime selection is gated by env SPEECH_CORE_ORT_PROVIDER + provider
    # availability. No extra link libs are needed — the EP lives inside the
    # imported onnxruntime shared object (or its companion provider .so loaded
    # at runtime by ORT itself).
    if(SPEECH_CORE_WITH_CUDA)
        target_compile_definitions(speech_core_models PUBLIC SPEECH_CORE_WITH_CUDA)
        message(STATUS "speech_core_models: NVIDIA GPU EPs enabled (SPEECH_CORE_WITH_CUDA). "
                       "Ensure ORT_DIR is a GPU-enabled ONNX Runtime.")
    endif()

    target_link_libraries(speech_core_models PUBLIC speech_core onnxruntime)
endif()

# ---------------------------------------------------------------------------
# speech_core_models_litert — LiteRT (TFLite) reference implementations (optional)
# ---------------------------------------------------------------------------

if(SPEECH_CORE_WITH_LITERT)
    if(NOT DEFINED LITERT_DIR)
        message(FATAL_ERROR "SPEECH_CORE_WITH_LITERT=ON requires LITERT_DIR — a directory containing libLiteRt.{so,dylib,dll} (extract from the ai-edge-litert PyPI wheel via scripts/fetch_litert.sh, or use the path printed by the script). Headers come from third_party/litert/ in the repo, not from LITERT_DIR.")
    endif()
    get_filename_component(LITERT_DIR "${LITERT_DIR}" ABSOLUTE)

    # libLiteRt from Google's ai-edge-litert package — imported shared library.
    # Windows needs both .dll (runtime) and .lib (link-time import library);
    # other platforms only need the shared object itself.
    add_library(litert SHARED IMPORTED)
    if(WIN32)
        set(_LITERT_LIB    "${LITERT_DIR}/libLiteRt.dll")
        set(_LITERT_IMPLIB "${LITERT_DIR}/libLiteRt.lib")
    elseif(APPLE)
        set(_LITERT_LIB "${LITERT_DIR}/libLiteRt.dylib")
    elseif(ANDROID)
        set(_LITERT_LIB "${LITERT_DIR}/${ANDROID_ABI}/libLiteRt.so")
    else()
        set(_LITERT_LIB "${LITERT_DIR}/libLiteRt.so")
    endif()
    set_target_properties(litert PROPERTIES
        IMPORTED_LOCATION "${_LITERT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert"
    )
    if(WIN32)
        set_target_properties(litert PROPERTIES IMPORTED_IMPLIB "${_LITERT_IMPLIB}")
    endif()

    add_library(speech_core_models_litert
        src/models/litert/litert_silero_vad.cpp
        src/models/litert/litert_parakeet_stt.cpp
        src/models/litert/litert_voxcpm2_tts.cpp
        src/models/litert/voxcpm2_c.cpp
        src/models/litert/voxcpm2_tokenizer.cpp
        src/models/litert/litert_wespeaker_embedding.cpp
        src/models/litert/litert_pyannote_segmentation.cpp
        src/models/litert/litert_omnilingual_stt.cpp
        src/models/litert/litert_nemotron_streaming_stt.cpp
    )

    target_include_directories(speech_core_models_litert
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )

    if(MSVC)
        target_compile_options(speech_core_models_litert PRIVATE /W4)
        target_compile_definitions(speech_core_models_litert PRIVATE _USE_MATH_DEFINES NOMINMAX)
    else()
        target_compile_options(speech_core_models_litert PRIVATE
            -Wall -Wextra
            $<$<CONFIG:Release>:-O2>
        )
    endif()

    set_target_properties(speech_core_models_litert PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(speech_core_models_litert PUBLIC speech_core litert)

    # VoxCPM2 voice-cloning CLI — desktop only (skipped on Android/embedded
    # consumers that link the library but don't want an executable target).
    if(NOT ANDROID)
        add_executable(speech_voxcpm2_clone examples/litert/voxcpm2_clone.cpp)
        target_link_libraries(speech_voxcpm2_clone PRIVATE speech_core_models_litert)
        if(MSVC)
            target_compile_definitions(speech_voxcpm2_clone PRIVATE _USE_MATH_DEFINES NOMINMAX)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if(SPEECH_CORE_BUILD_TESTS)
    enable_testing()

    # Orchestration tests — always built, link speech_core only.
    file(GLOB TEST_SOURCES tests/test_*.cpp)
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        # test_models.cpp needs ORT — handled below when SPEECH_CORE_WITH_ONNX=ON.
        # test_litert_models.cpp needs LiteRT — handled below when SPEECH_CORE_WITH_LITERT=ON.
        if(TEST_NAME STREQUAL "test_models" OR TEST_NAME STREQUAL "test_litert_models")
            continue()
        endif()
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE speech_core)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()

    # Model integration tests — require SPEECH_CORE_WITH_ONNX=ON.
    # Skip gracefully at runtime when SPEECH_MODEL_DIR isn't set.
    if(SPEECH_CORE_WITH_ONNX AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_models.cpp)
        add_executable(test_models tests/test_models.cpp)
        target_link_libraries(test_models PRIVATE speech_core_models)
        target_compile_definitions(test_models PRIVATE
            SPEECH_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
        add_test(NAME test_models COMMAND test_models)

        # ORT micro-benchmark (perf tool, not registered as a ctest). Picks
        # CPU vs CUDA via SPEECH_CORE_ORT_PROVIDER at runtime, so the same
        # binary serves both reports.
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_ort_models.cpp)
            add_executable(bench_ort_models tests/bench_ort_models.cpp)
            target_link_libraries(bench_ort_models PRIVATE speech_core_models)
            target_compile_definitions(bench_ort_models PRIVATE
                SPEECH_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
        endif()
    endif()

    # LiteRT model integration tests — require SPEECH_CORE_WITH_LITERT=ON.
    # Skip gracefully at runtime when SPEECH_LITERT_MODEL_DIR isn't set.
    if(SPEECH_CORE_WITH_LITERT AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_litert_models.cpp)
        add_executable(test_litert_models tests/test_litert_models.cpp)
        target_link_libraries(test_litert_models PRIVATE speech_core_models_litert)
        target_compile_definitions(test_litert_models PRIVATE
            SPEECH_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
        add_test(NAME test_litert_models COMMAND test_litert_models)

        # CPU STT micro-benchmark (not registered as a ctest — perf tool only).
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_litert_stt.cpp)
            add_executable(bench_litert_stt tests/bench_litert_stt.cpp)
            target_link_libraries(bench_litert_stt PRIVATE speech_core_models_litert)
            target_compile_definitions(bench_litert_stt PRIVATE
                SPEECH_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

set(_install_targets speech_core)
if(SPEECH_CORE_WITH_ONNX)
    list(APPEND _install_targets speech_core_models)
endif()
if(SPEECH_CORE_WITH_LITERT)
    list(APPEND _install_targets speech_core_models_litert)
endif()

install(TARGETS ${_install_targets}
    EXPORT speech_core-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/speech_core
    DESTINATION include
)

# ---------------------------------------------------------------------------
# Examples (optional)
# ---------------------------------------------------------------------------

if(SPEECH_CORE_BUILD_EXAMPLES)
    if(NOT SPEECH_CORE_WITH_ONNX)
        message(FATAL_ERROR
            "SPEECH_CORE_BUILD_EXAMPLES=ON requires SPEECH_CORE_WITH_ONNX=ON"
            " (the examples consume speech_core_models)")
    endif()
    add_subdirectory(examples/linux)
endif()
