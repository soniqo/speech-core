cmake_minimum_required(VERSION 3.16)
project(speech_linux_example VERSION 0.0.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SPEECH_LINUX_BUILD_DEMO "Build the ALSA demo CLI" ON)
option(SPEECH_LINUX_BUILD_TOOLS "Build the transcribe/synthesize/phonemize CLIs" ON)
option(SPEECH_LINUX_BUILD_TESTS "Build the Linux integration test" ON)

# ---------------------------------------------------------------------------
# speech_core + speech_core_models
#
# When invoked as a subdirectory of the top-level speech-core build (via
# SPEECH_CORE_BUILD_EXAMPLES=ON), the speech_core / speech_core_models targets
# already exist and are reused. When invoked standalone, we resolve them via
# find_package or fall back to add_subdirectory of the parent repo.
# ---------------------------------------------------------------------------

if(NOT TARGET speech_core_models)
    # Standalone build: pull in the parent repo. Caller must set
    # SPEECH_CORE_WITH_ONNX=ON and ORT_DIR before invoking us.
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. speech_core)
endif()

if(NOT TARGET speech_core_models)
    message(FATAL_ERROR
        "examples/linux requires speech_core_models. Configure the parent build with"
        " -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=...")
endif()

# ---------------------------------------------------------------------------
# libspeech.so — small C ABI wrapping speech_core::VoicePipeline
# ---------------------------------------------------------------------------

add_library(speech SHARED src/speech.cpp)

target_include_directories(speech
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(speech PUBLIC speech_core_models)

target_compile_options(speech PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Release>:-O2>
)

# ---------------------------------------------------------------------------
# Demo CLI — full conversational loop driven by ALSA mic input
# ---------------------------------------------------------------------------

if(SPEECH_LINUX_BUILD_DEMO)
    add_executable(speech_demo demo/main.cpp)
    target_link_libraries(speech_demo PRIVATE speech)

    find_library(ALSA_LIB asound)
    if(ALSA_LIB)
        target_link_libraries(speech_demo PRIVATE ${ALSA_LIB})
        target_compile_definitions(speech_demo PRIVATE HAS_ALSA=1)
    else()
        message(STATUS "ALSA not found — speech_demo will be a no-op stub")
    endif()
endif()

# ---------------------------------------------------------------------------
# CLI tools — direct model use (no pipeline)
# ---------------------------------------------------------------------------

if(SPEECH_LINUX_BUILD_TOOLS)
    # speech_transcribe uses the C ABI (speech.h) — link through libspeech.
    add_executable(speech_transcribe tools/transcribe.cpp)
    target_link_libraries(speech_transcribe PRIVATE speech)

    # speech_synthesize and speech_phonemize call models directly.
    add_executable(speech_synthesize tools/synthesize.cpp)
    target_link_libraries(speech_synthesize PRIVATE speech_core_models)

    add_executable(speech_phonemize tools/phonemize.cpp)
    target_link_libraries(speech_phonemize PRIVATE speech_core_models)
endif()

# ---------------------------------------------------------------------------
# Integration test — exercises the C ABI end-to-end with real models
# Runs under ctest only when SPEECH_MODEL_DIR is set (skips otherwise).
# ---------------------------------------------------------------------------

if(SPEECH_LINUX_BUILD_TESTS)
    add_executable(speech_linux_test tests/test_pipeline.cpp)
    target_link_libraries(speech_linux_test PRIVATE speech)

    enable_testing()
    add_test(NAME speech_linux_pipeline COMMAND speech_linux_test)
endif()
