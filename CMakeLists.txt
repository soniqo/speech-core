cmake_minimum_required(VERSION 3.16)
project(speech_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(SPEECH_CORE_BUILD_TESTS "Build tests" ON)
option(SPEECH_CORE_WITH_ONNX "Build the speech_core_models target with ONNX Runtime reference implementations (Silero VAD, Parakeet STT, Kokoro TTS, DeepFilter)" OFF)

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
    src/speech_core_c.cpp
)

target_include_directories(speech_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(speech_core PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O2>
)

# ---------------------------------------------------------------------------
# speech_core_models — ONNX Runtime reference implementations (optional)
# ---------------------------------------------------------------------------

if(SPEECH_CORE_WITH_ONNX)
    if(NOT DEFINED ORT_DIR)
        message(FATAL_ERROR "SPEECH_CORE_WITH_ONNX=ON requires ORT_DIR (path to ONNX Runtime with include/ and lib/)")
    endif()

    # ONNX Runtime — imported shared library
    add_library(onnxruntime SHARED IMPORTED)
    if(APPLE)
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

    target_compile_options(speech_core_models PRIVATE
        -Wall -Wextra
        $<$<CONFIG:Release>:-O2>
    )

    target_link_libraries(speech_core_models PUBLIC speech_core onnxruntime)
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
        if(TEST_NAME STREQUAL "test_models")
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
    endif()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

set(_install_targets speech_core)
if(SPEECH_CORE_WITH_ONNX)
    list(APPEND _install_targets speech_core_models)
endif()

install(TARGETS ${_install_targets}
    EXPORT speech_core-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/speech_core
    DESTINATION include
)
