cmake_minimum_required(VERSION 3.16)
project(speech_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(SPEECH_CORE_BUILD_TESTS "Build tests" ON)
option(SPEECH_CORE_WITH_ONNX "Build the speech_core_models target with ONNX Runtime reference implementations (Silero VAD, Parakeet STT, Kokoro TTS, DeepFilter)." OFF)
option(SPEECH_CORE_WITH_LITERT "Build the speech_core_models_litert target with LiteRT (TFLite) reference implementations (Silero VAD, Parakeet STT)" OFF)
option(SPEECH_CORE_WITH_HF_DOWNLOAD "Enable libcurl-backed Hugging Face model downloads (sc_voxcpm2_create_from_pretrained: resumable, retried first-run fetch of the LiteRT bundle). Requires SPEECH_CORE_WITH_LITERT=ON and libcurl (find_package(CURL))." OFF)
option(SPEECH_CORE_BUILD_EXAMPLES "Build examples/ (Linux C ABI lib, ALSA demo, CLI tools, integration tests). Requires SPEECH_CORE_WITH_ONNX=ON." OFF)
option(SPEECH_CORE_WITH_OLLAMA "Build the speech_core_llm_ollama target — Ollama HTTP LLM adapter (tool-calling agents over /api/chat). Vendors cpp-httplib + nlohmann/json under third_party/; no system deps beyond OS sockets." OFF)
option(SPEECH_CORE_PACKAGE "Add install rules for the CLI tools + bundled runtime libraries and configure CPack (Linux .deb / .tar.gz). Linux-only; see the Packaging section at the bottom of this file." OFF)
set(SPEECH_CORE_PACKAGE_VERSION "${PROJECT_VERSION}" CACHE STRING "Version stamped into binary packages. The release workflow passes the git tag (e.g. 0.0.8); local builds default to the project version.")

if(SPEECH_CORE_WITH_HF_DOWNLOAD AND NOT SPEECH_CORE_WITH_LITERT)
    message(FATAL_ERROR "SPEECH_CORE_WITH_HF_DOWNLOAD=ON requires SPEECH_CORE_WITH_LITERT=ON "
                        "(the download feature fetches the LiteRT VoxCPM2 bundle).")
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
    src/audio/seamless_fbank.cpp
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

    set(_SPEECH_CORE_MODELS_SOURCES
        src/models/silero/silero_vad.cpp
        src/models/parakeet/parakeet_stt.cpp
        src/models/nemotron/nemotron_multilingual_stt.cpp
        src/models/kokoro/kokoro_tts.cpp
        src/models/kokoro/kokoro_phonemizer.cpp
        src/models/kokoro/kokoro_multilingual.cpp
        src/models/deepfilter/deepfilter.cpp
        src/models/sidon/onnx_sidon_restorer.cpp
        src/models/voxcpm2/onnx_voxcpm2_tts.cpp
        src/models/nemotron/onnx_nemotron_streaming_stt.cpp
        src/models/personaplex/onnx_personaplex.cpp
    )
    # VoxCPM2 tokenizer is pure C++17 and shared with the LiteRT target. To
    # avoid duplicate symbols when a downstream binary links both targets, we
    # only compile it into speech_core_models when the LiteRT target isn't
    # being built. When both are on, speech_core_models_litert owns the symbol
    # and speech_core_models picks it up through the link line.
    if(NOT SPEECH_CORE_WITH_LITERT)
        list(APPEND _SPEECH_CORE_MODELS_SOURCES
            src/models/litert/voxcpm2_tokenizer.cpp)
    endif()
    add_library(speech_core_models ${_SPEECH_CORE_MODELS_SOURCES})

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

    target_link_libraries(speech_core_models PUBLIC speech_core onnxruntime)

    # VoxCPM2 ONNX voice-cloning CLI — the in-repo runner for the ONNX path:
    # the full 4-graph AR loop through OnnxVoxCPM2Tts, on CPU or CUDA/TensorRT
    # (SPEECH_CORE_ORT_PROVIDER). Desktop only; mirrors the LiteRT
    # speech_voxcpm2_clone so the two backends validate identically.
    if(NOT ANDROID)
        add_executable(speech_voxcpm2_clone_onnx examples/onnx/voxcpm2_clone.cpp)
        target_link_libraries(speech_voxcpm2_clone_onnx PRIVATE speech_core_models)
        # VoxCPM2Tokenizer is compiled into speech_core_models_litert when both
        # backends build (see the NOT SPEECH_CORE_WITH_LITERT guard above);
        # link it through so the ONNX CLI resolves the tokenizer symbols.
        if(SPEECH_CORE_WITH_LITERT)
            target_link_libraries(speech_voxcpm2_clone_onnx PRIVATE speech_core_models_litert)
        endif()
        if(MSVC)
            target_compile_definitions(speech_voxcpm2_clone_onnx PRIVATE _USE_MATH_DEFINES NOMINMAX)
        endif()

        # Sidon speech-restoration CLI (ONNX): denoise + dereverb a clip to
        # 48 kHz. The in-repo runner for validating the predictor + DAC vocoder
        # pipeline (and the C++ SeamlessM4T front-end) end-to-end. Desktop only.
        add_executable(speech_sidon_restore examples/onnx/sidon_restore.cpp)
        target_link_libraries(speech_sidon_restore PRIVATE speech_core_models)
        if(MSVC)
            target_compile_definitions(speech_sidon_restore PRIVATE _USE_MATH_DEFINES NOMINMAX)
        endif()
    endif()
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
        src/models/litert/hf_download.cpp
        src/models/litert/voxcpm2_tokenizer.cpp
        src/models/litert/chatterbox_tokenizer.cpp
        src/models/litert/litert_chatterbox_tts.cpp
        src/models/litert/chatterbox_c.cpp
        src/models/litert/litert_wespeaker_embedding.cpp
        src/models/litert/litert_pyannote_segmentation.cpp
        src/models/litert/litert_omnilingual_stt.cpp
        src/models/litert/litert_nemotron_streaming_stt.cpp
        src/models/litert/litert_nemotron_multilingual_stt.cpp
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

    # Optional first-run model download (sc_voxcpm2_create_from_pretrained).
    # libcurl provides resumable HTTPS with an OS-native TLS backend (Schannel
    # on Windows, Secure Transport / OpenSSL elsewhere). PRIVATE: the C ABI
    # hides curl entirely, so downstreams don't need libcurl headers.
    if(SPEECH_CORE_WITH_HF_DOWNLOAD)
        find_package(CURL REQUIRED)
        target_link_libraries(speech_core_models_litert PRIVATE CURL::libcurl)
        target_compile_definitions(speech_core_models_litert PRIVATE SPEECH_CORE_WITH_HF_DOWNLOAD)
    endif()

    # VoxCPM2 voice-cloning CLI — desktop only (skipped on Android/embedded
    # consumers that link the library but don't want an executable target).
    if(NOT ANDROID)
        add_executable(speech_voxcpm2_clone examples/litert/voxcpm2_clone.cpp)
        target_link_libraries(speech_voxcpm2_clone PRIVATE speech_core_models_litert)
        if(MSVC)
            target_compile_definitions(speech_voxcpm2_clone PRIVATE _USE_MATH_DEFINES NOMINMAX)
        endif()

        # Chatterbox T3 LiteRT loop verification driver (prefill+step vs golden).
        add_executable(chatterbox_t3_check examples/litert/chatterbox_t3_check.cpp)
        target_link_libraries(chatterbox_t3_check PRIVATE speech_core_models_litert)
        if(MSVC)
            target_compile_definitions(chatterbox_t3_check PRIVATE _USE_MATH_DEFINES NOMINMAX)
        endif()

        # Chatterbox flow-path verification driver (encoder_v2 -> Euler/CFG -> hift vs golden).
        add_executable(chatterbox_flow_check examples/litert/chatterbox_flow_check.cpp)
        target_link_libraries(chatterbox_flow_check PRIVATE speech_core_models_litert)
        if(MSVC)
            target_compile_definitions(chatterbox_flow_check PRIVATE _USE_MATH_DEFINES NOMINMAX)
        endif()

        # Chatterbox tokenizer validation vs Python golden IDs.
        add_executable(chatterbox_tok_check examples/litert/chatterbox_tok_check.cpp)
        target_link_libraries(chatterbox_tok_check PRIVATE speech_core_models_litert)
        if(MSVC)
            target_compile_definitions(chatterbox_tok_check PRIVATE _USE_MATH_DEFINES NOMINMAX)
        endif()

        # Chatterbox TTS CLI: text -> 24 kHz wav via the full wrapper.
        add_executable(chatterbox_tts_cli examples/litert/chatterbox_tts_cli.cpp)
        target_link_libraries(chatterbox_tts_cli PRIVATE speech_core_models_litert)
        if(MSVC)
            target_compile_definitions(chatterbox_tts_cli PRIVATE _USE_MATH_DEFINES NOMINMAX)
        endif()
    endif()

    # Debug CLI for the HF downloader (resume/retry testing). Links the internal
    # hf_download header directly; libcurl rides in via speech_core_models_litert.
    if(SPEECH_CORE_WITH_HF_DOWNLOAD AND NOT ANDROID)
        add_executable(hf_fetch examples/litert/hf_fetch.cpp)
        target_link_libraries(hf_fetch PRIVATE speech_core_models_litert)
        target_include_directories(hf_fetch PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/models/litert)
        if(MSVC)
            target_compile_definitions(hf_fetch PRIVATE _USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# speech_core_llm_ollama — Ollama HTTP LLM adapter (optional)
# ---------------------------------------------------------------------------

if(SPEECH_CORE_WITH_OLLAMA)
    add_library(speech_core_llm_ollama
        src/llm/ollama_llm.cpp
    )

    target_include_directories(speech_core_llm_ollama
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )

    if(MSVC)
        target_compile_options(speech_core_llm_ollama PRIVATE /W4)
        target_compile_definitions(speech_core_llm_ollama PRIVATE
            _USE_MATH_DEFINES NOMINMAX _WIN32_WINNT=0x0601)
    else()
        target_compile_options(speech_core_llm_ollama PRIVATE
            -Wall -Wextra
            $<$<CONFIG:Release>:-O2>
        )
    endif()

    set_target_properties(speech_core_llm_ollama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(speech_core_llm_ollama PUBLIC speech_core)

    # Windows: cpp-httplib needs WinSock 2.
    if(WIN32)
        target_link_libraries(speech_core_llm_ollama PRIVATE ws2_32)
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
        # test_litert_models.cpp / test_litert_voxcpm2_clone_cli.cpp need
        # LiteRT — handled below when SPEECH_CORE_WITH_LITERT=ON.
        if(TEST_NAME STREQUAL "test_models" OR TEST_NAME STREQUAL "test_litert_models"
           OR TEST_NAME STREQUAL "test_litert_chatterbox"
           OR TEST_NAME STREQUAL "test_hf_download"
           OR TEST_NAME STREQUAL "test_litert_voxcpm2_clone_cli"
           OR TEST_NAME STREQUAL "test_ollama_llm"
           OR TEST_NAME STREQUAL "test_pipeline_ollama_e2e"
           OR TEST_NAME STREQUAL "test_fdb_bench_smoke"
           OR TEST_NAME STREQUAL "test_fdb_summary")
            continue()
        endif()
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE speech_core)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()

    # HF downloader decision-logic test — the pure resume/skip/restart rules
    # behind sc_voxcpm2_create_from_pretrained. Needs no libcurl or LiteRT, so it
    # compiles hf_download.cpp's no-network branch directly and runs in every
    # configuration (not just LiteRT builds).
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hf_download.cpp)
        add_executable(test_hf_download
            tests/test_hf_download.cpp
            src/models/litert/hf_download.cpp)
        target_include_directories(test_hf_download PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/models/litert
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
        add_test(NAME test_hf_download COMMAND test_hf_download)
    endif()

    # Model integration tests — require SPEECH_CORE_WITH_ONNX=ON.
    # Skip gracefully at runtime when SPEECH_MODEL_DIR isn't set.
    if(SPEECH_CORE_WITH_ONNX AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_models.cpp)
        add_executable(test_models tests/test_models.cpp)
        target_link_libraries(test_models PRIVATE speech_core_models)
        if(SPEECH_CORE_WITH_LITERT)
            target_link_libraries(test_models PRIVATE speech_core_models_litert)
        endif()
        target_compile_definitions(test_models PRIVATE
            SPEECH_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
        add_test(NAME test_models COMMAND test_models)

        # PersonaPlex standalone CLI smoke harness — load the wrapper against
        # an on-disk bundle and run respond_stream. Not a ctest (interactive
        # tool); built alongside test_models when the .cpp is present.
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_personaplex.cpp)
            add_executable(run_personaplex tests/run_personaplex.cpp)
            target_link_libraries(run_personaplex PRIVATE speech_core_models)
        endif()

        # ORT micro-benchmark (perf tool, not registered as a ctest). Picks
        # CPU vs CUDA via SPEECH_CORE_ORT_PROVIDER at runtime, so the same
        # binary serves both reports.
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_ort_models.cpp)
            add_executable(bench_ort_models tests/bench_ort_models.cpp)
            target_link_libraries(bench_ort_models PRIVATE speech_core_models)
            if(SPEECH_CORE_WITH_LITERT)
                # Tokenizer symbol lives in speech_core_models_litert when both
                # backends are enabled (see the conditional above the sources
                # list) — link it so OnnxVoxCPM2Tts's references resolve.
                target_link_libraries(bench_ort_models PRIVATE speech_core_models_litert)
            endif()
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
    endif()

    # Chatterbox TTS smoke test (skips when SPEECH_CORE_CHATTERBOX_BUNDLE unset).
    if(SPEECH_CORE_WITH_LITERT AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_litert_chatterbox.cpp)
        add_executable(test_litert_chatterbox tests/test_litert_chatterbox.cpp)
        target_link_libraries(test_litert_chatterbox PRIVATE speech_core_models_litert)
        add_test(NAME test_litert_chatterbox COMMAND test_litert_chatterbox)

        # CLI-boundary tests for speech_voxcpm2_clone — these spawn the real
        # executable so OS argv encoding is part of the test surface (a
        # Windows active-code-page conversion once turned a Devanagari text
        # argument into '?', invisible to every in-process test). The ASR
        # roundtrip layer skips unless SPEECH_LITERT_MODEL_DIR is set.
        if(NOT ANDROID)
            add_executable(test_litert_voxcpm2_clone_cli tests/test_litert_voxcpm2_clone_cli.cpp)
            target_link_libraries(test_litert_voxcpm2_clone_cli PRIVATE speech_core_models_litert)
            target_compile_definitions(test_litert_voxcpm2_clone_cli PRIVATE
                SPEECH_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
            add_dependencies(test_litert_voxcpm2_clone_cli speech_voxcpm2_clone)
            add_test(NAME test_litert_voxcpm2_clone_cli COMMAND test_litert_voxcpm2_clone_cli)
        endif()

        # CPU STT micro-benchmark (not registered as a ctest — perf tool only).
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_litert_stt.cpp)
            add_executable(bench_litert_stt tests/bench_litert_stt.cpp)
            target_link_libraries(bench_litert_stt PRIVATE speech_core_models_litert)
            target_compile_definitions(bench_litert_stt PRIVATE
                SPEECH_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
        endif()
    endif()

    # Ollama LLM unit + opt-in integration test — runs against an in-process
    # httplib mock server, no real Ollama required for the default suite.
    if(SPEECH_CORE_WITH_OLLAMA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ollama_llm.cpp)
        add_executable(test_ollama_llm tests/test_ollama_llm.cpp)
        target_link_libraries(test_ollama_llm PRIVATE speech_core_llm_ollama)
        target_include_directories(test_ollama_llm PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
        if(MSVC)
            target_compile_definitions(test_ollama_llm PRIVATE
                _USE_MATH_DEFINES NOMINMAX _WIN32_WINNT=0x0601)
        endif()
        if(WIN32)
            target_link_libraries(test_ollama_llm PRIVATE ws2_32)
        endif()
        add_test(NAME test_ollama_llm COMMAND test_ollama_llm)
    endif()

    # End-to-end test that wires the real OllamaLLM adapter into
    # VoicePipeline against an in-process httplib mock server. Proves the
    # full STT -> LLM -> TTS path (including barge-in cancel + tool-loop)
    # works with the real Ollama adapter rather than only the MockLLM in
    # test_pipeline_e2e. Links both speech_core (pipeline) and
    # speech_core_llm_ollama (LLM adapter). Same gate as test_ollama_llm.
    if(SPEECH_CORE_WITH_OLLAMA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pipeline_ollama_e2e.cpp)
        add_executable(test_pipeline_ollama_e2e tests/test_pipeline_ollama_e2e.cpp)
        target_link_libraries(test_pipeline_ollama_e2e
            PRIVATE speech_core speech_core_llm_ollama)
        target_include_directories(test_pipeline_ollama_e2e PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
        if(MSVC)
            target_compile_definitions(test_pipeline_ollama_e2e PRIVATE
                _USE_MATH_DEFINES NOMINMAX _WIN32_WINNT=0x0601)
        endif()
        if(WIN32)
            target_link_libraries(test_pipeline_ollama_e2e PRIVATE ws2_32)
        endif()
        add_test(NAME test_pipeline_ollama_e2e COMMAND test_pipeline_ollama_e2e)
    endif()

    # FDB v1.0 per-sample driver — corpus iterator + WAV I/O + driver
    # skeleton (M1). Default backends are mock STT + mock TTS so the
    # binary builds and runs without ORT or model files. --stt parakeet
    # / --tts kokoro switch to real backends when SPEECH_CORE_WITH_ONNX
    # is also ON (compile flag SPEECH_CORE_FDB_WITH_ONNX gates the
    # branch inside fdb_bench.cpp). The LLM is always OllamaLLM.
    if(SPEECH_CORE_WITH_OLLAMA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/fdb_bench/fdb_bench.cpp)
        add_executable(fdb_bench
            examples/fdb_bench/fdb_bench.cpp
            examples/fdb_bench/fdb_corpus.cpp
            examples/fdb_bench/wav_io.cpp
        )
        target_include_directories(fdb_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/fdb_bench)
        target_link_libraries(fdb_bench PRIVATE
            speech_core speech_core_llm_ollama)
        if(SPEECH_CORE_WITH_ONNX)
            target_link_libraries(fdb_bench PRIVATE speech_core_models)
            target_compile_definitions(fdb_bench PRIVATE SPEECH_CORE_FDB_WITH_ONNX)
        endif()
        if(MSVC)
            target_compile_definitions(fdb_bench PRIVATE
                _USE_MATH_DEFINES NOMINMAX _WIN32_WINNT=0x0601)
        endif()
        if(WIN32)
            target_link_libraries(fdb_bench PRIVATE ws2_32)
        endif()

        # Smoke test — runs the driver's per-sample loop against the
        # tiny tests/data/fdb_mini fixture with mock STT/TTS + an
        # in-process MockOllama. No model files, no live network.
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fdb_bench_smoke.cpp)
            add_executable(test_fdb_bench_smoke
                tests/test_fdb_bench_smoke.cpp
                examples/fdb_bench/fdb_corpus.cpp
                examples/fdb_bench/wav_io.cpp)
            target_include_directories(test_fdb_bench_smoke PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/examples/fdb_bench
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
            target_link_libraries(test_fdb_bench_smoke
                PRIVATE speech_core speech_core_llm_ollama)
            target_compile_definitions(test_fdb_bench_smoke PRIVATE
                SPEECH_CORE_FDB_MINI_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/fdb_mini/v1_0")
            if(SPEECH_CORE_WITH_ONNX)
                target_link_libraries(test_fdb_bench_smoke
                    PRIVATE speech_core_models)
                target_compile_definitions(test_fdb_bench_smoke PRIVATE
                    SPEECH_CORE_FDB_WITH_ONNX
                    SPEECH_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
            endif()
            if(MSVC)
                target_compile_definitions(test_fdb_bench_smoke PRIVATE
                    _USE_MATH_DEFINES NOMINMAX _WIN32_WINNT=0x0601)
            endif()
            if(WIN32)
                target_link_libraries(test_fdb_bench_smoke PRIVATE ws2_32)
            endif()
            add_test(NAME test_fdb_bench_smoke COMMAND test_fdb_bench_smoke)
        endif()
    endif()

    # fdb_summary — M3 post-processor that reads <out-dir>/*.json from
    # fdb_bench and emits a single per-bucket CSV (p50/p90/p99 of
    # stt/llm/tts/ttft/total). Gated identically to fdb_bench.
    if(SPEECH_CORE_WITH_OLLAMA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/fdb_bench/fdb_summary.cpp)
        add_executable(fdb_summary
            examples/fdb_bench/fdb_summary.cpp
            examples/fdb_bench/fdb_summary_lib.cpp
        )
        target_include_directories(fdb_summary PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/fdb_bench
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
        target_link_libraries(fdb_summary PRIVATE speech_core)
        if(MSVC)
            target_compile_definitions(fdb_summary PRIVATE
                _USE_MATH_DEFINES NOMINMAX _WIN32_WINNT=0x0601)
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fdb_summary.cpp)
            add_executable(test_fdb_summary
                tests/test_fdb_summary.cpp
                examples/fdb_bench/fdb_summary_lib.cpp)
            target_include_directories(test_fdb_summary PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/examples/fdb_bench
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
            target_link_libraries(test_fdb_summary PRIVATE speech_core)
            if(MSVC)
                target_compile_definitions(test_fdb_summary PRIVATE
                    _USE_MATH_DEFINES NOMINMAX _WIN32_WINNT=0x0601)
            endif()
            add_test(NAME test_fdb_summary COMMAND test_fdb_summary)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

# Dev install rules (static libs + headers) — skipped when building the
# end-user CLI package (SPEECH_CORE_PACKAGE=ON), which would otherwise drop
# tens of MB of .a files and the header tree into /usr.
if(NOT SPEECH_CORE_PACKAGE)
    set(_install_targets speech_core)
    if(SPEECH_CORE_WITH_ONNX)
        list(APPEND _install_targets speech_core_models)
    endif()
    if(SPEECH_CORE_WITH_LITERT)
        list(APPEND _install_targets speech_core_models_litert)
    endif()
    if(SPEECH_CORE_WITH_OLLAMA)
        list(APPEND _install_targets speech_core_llm_ollama)
    endif()

    install(TARGETS ${_install_targets}
        EXPORT speech_core-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )

    install(DIRECTORY include/speech_core
        DESTINATION include
    )
endif()

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

# ---------------------------------------------------------------------------
# Packaging (optional) — Linux .deb / .tar.gz with the CLI tools
#
# Adds RUNTIME install rules for whichever CLI executables the enabled options
# produced, bundles the runtime shared libraries that no distro packages
# (libonnxruntime.so, libLiteRt.so) into a private lib/speech-core/ dir wired
# up via $ORIGIN RPATHs, and configures CPack DEB+TGZ generators. Models are
# never packaged — they are fetched at runtime from HuggingFace (see
# scripts/download_models*.sh and docs/models.md).
#
# Driven by .github/workflows/release.yml on v* tags; for a local package:
#   cmake -B build-pkg -DCMAKE_BUILD_TYPE=Release \
#       -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=ort-linux \
#       -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=litert-linux \
#       -DSPEECH_CORE_BUILD_EXAMPLES=ON -DSPEECH_CORE_BUILD_TESTS=OFF \
#       -DSPEECH_CORE_PACKAGE=ON
#   cmake --build build-pkg --parallel && (cd build-pkg && cpack -B ../dist)
# ---------------------------------------------------------------------------

if(SPEECH_CORE_PACKAGE)
    if(NOT UNIX OR APPLE OR ANDROID)
        message(FATAL_ERROR "SPEECH_CORE_PACKAGE=ON currently targets Linux only")
    endif()

    # Collect whichever CLI targets this configuration actually built.
    set(_pkg_cli_targets "")
    foreach(_t speech_demo speech_transcribe speech_synthesize speech_phonemize
               speech_voxcpm2_clone)
        if(TARGET ${_t})
            list(APPEND _pkg_cli_targets ${_t})
        endif()
    endforeach()
    if(NOT _pkg_cli_targets)
        message(FATAL_ERROR
            "SPEECH_CORE_PACKAGE=ON but no CLI executables are enabled — turn on"
            " SPEECH_CORE_BUILD_EXAMPLES (ONNX tools) and/or SPEECH_CORE_WITH_LITERT"
            " (speech_voxcpm2_clone)")
    endif()

    # Executables install to bin/, bundled runtimes to lib/speech/, the
    # example C ABI lib (libspeech.so) to lib/. $ORIGIN RPATHs make the package
    # self-contained without ld.so.conf entries.
    set_target_properties(${_pkg_cli_targets} PROPERTIES
        INSTALL_RPATH "\$ORIGIN/../lib;\$ORIGIN/../lib/speech")
    install(TARGETS ${_pkg_cli_targets} RUNTIME DESTINATION bin)

    if(TARGET speech)
        set_target_properties(speech PROPERTIES
            INSTALL_RPATH "\$ORIGIN;\$ORIGIN/speech")
        install(TARGETS speech LIBRARY DESTINATION lib)
    endif()

    # Runtime shared libraries that no distro packages. setup_linux.sh keeps
    # the libonnxruntime.so -> .so.X.Y.Z symlink chain (cp -RP); the glob picks
    # up every name and install(PROGRAMS) preserves symlinks, so the SONAME
    # chain stays intact in lib/speech without duplicate payload.
    if(SPEECH_CORE_WITH_ONNX)
        file(GLOB _pkg_ort_libs "${ORT_DIR}/lib/libonnxruntime.so*")
        if(NOT _pkg_ort_libs)
            message(FATAL_ERROR "SPEECH_CORE_PACKAGE: no libonnxruntime.so* under ${ORT_DIR}/lib")
        endif()
        install(PROGRAMS ${_pkg_ort_libs} DESTINATION lib/speech)
    endif()
    if(SPEECH_CORE_WITH_LITERT)
        install(PROGRAMS "${LITERT_DIR}/libLiteRt.so" DESTINATION lib/speech)
    endif()

    install(FILES LICENSE THIRD-PARTY-NOTICES.md DESTINATION share/doc/speech)

    # Model fetchers, renamed into the speech_ command namespace. When run
    # from the read-only install location they default to the per-user cache
    # dir (~/.cache/speech-core/models*) — the same place the CLIs look when
    # invoked without a model_dir argument.
    install(PROGRAMS scripts/download_models.sh
        DESTINATION bin RENAME speech_download_models)
    if(SPEECH_CORE_WITH_LITERT)
        install(PROGRAMS scripts/download_models_litert.sh
            DESTINATION bin RENAME speech_download_models_litert)
    endif()

    # `speech` multiplexer — verb-style front-end (speech transcribe …,
    # speech speak …) matching speech-swift's CLI on macOS
    # (brew install soniqo/tap/speech), so cross-platform docs share one
    # command surface. Dispatches to the sibling speech_* binaries.
    install(PROGRAMS scripts/speech-dispatch.sh DESTINATION bin RENAME speech)

    # --- CPack ---
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_pkg_arch "arm64")
    else()
        set(_pkg_arch "amd64")
    endif()

    set(CPACK_GENERATOR "DEB;TGZ")
    set(CPACK_PACKAGE_NAME "speech")
    set(CPACK_PACKAGE_VERSION "${SPEECH_CORE_PACKAGE_VERSION}")
    set(CPACK_PACKAGE_CONTACT "ivan-digital <ivan.aufkl@gmail.com>")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
        "speech — voice agent CLI tools (STT, TTS, phonemizer, voice cloning)")
    set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/soniqo/speech-core")
    set(CPACK_PACKAGE_FILE_NAME "speech-${SPEECH_CORE_PACKAGE_VERSION}-linux-${_pkg_arch}")
    set(CPACK_STRIP_FILES ON)
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "${_pkg_arch}")
    set(CPACK_DEBIAN_PACKAGE_SECTION "sound")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/soniqo/speech-core")
    # Hand-written Depends (dpkg-shlibdeps can't resolve our private
    # lib/speech-core dir). Built on ubuntu-22.04 → needs glibc ≥ 2.35.
    # libasound2t64 is the 24.04 time64 rename; the alternation covers 22.04.
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.35), libstdc++6 (>= 11)")
    if(TARGET speech_demo)
        string(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS ", libasound2t64 | libasound2")
    endif()

    include(CPack)
endif()
