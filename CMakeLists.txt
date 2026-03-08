cmake_minimum_required(VERSION 3.16)
project(speech_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library sources
set(SOURCES
    src/pipeline/voice_pipeline.cpp
    src/pipeline/turn_detector.cpp
    src/pipeline/speech_queue.cpp
    src/pipeline/conversation_context.cpp
    src/vad/streaming_vad.cpp
    src/audio/audio_buffer.cpp
    src/audio/resampler.cpp
    src/audio/pcm_codec.cpp
    src/protocol/realtime_protocol.cpp
    src/speech_core_c.cpp
)

add_library(speech_core ${SOURCES})

target_include_directories(speech_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(speech_core PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O2>
)

# Tests
option(SPEECH_CORE_BUILD_TESTS "Build tests" ON)

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
