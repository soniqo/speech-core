cmake_minimum_required(VERSION 3.16)
project(speech_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(SPEECH_CORE_BUILD_TESTS "Build tests" ON)
option(SPEECH_CORE_WITH_ONNX "Build ONNX Runtime model wrappers (Silero VAD, Parakeet STT, Kokoro TTS, DeepFilter)" OFF)

# Core sources (always built — orchestration + interfaces + audio utilities, no ML deps)
set(CORE_SOURCES
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

# ONNX Runtime model wrappers (optional)
set(MODEL_SOURCES
    src/models/silero/silero_vad.cpp
    src/models/parakeet/parakeet_stt.cpp
    src/models/kokoro/kokoro_tts.cpp
    src/models/kokoro/kokoro_phonemizer.cpp
    src/models/kokoro/kokoro_multilingual.cpp
    src/models/deepfilter/deepfilter.cpp
)

add_library(speech_core ${CORE_SOURCES})

target_include_directories(speech_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(speech_core PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O2>
)

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

    target_sources(speech_core PRIVATE ${MODEL_SOURCES})
    target_link_libraries(speech_core PUBLIC onnxruntime)
endif()

# Tests
if(SPEECH_CORE_BUILD_TESTS)
    enable_testing()

    file(GLOB TEST_SOURCES tests/test_*.cpp)
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE speech_core)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

# Install
install(TARGETS speech_core
    EXPORT speech_core-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/speech_core
    DESTINATION include
)
