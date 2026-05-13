# Reproducible Linux build of speech-core + the examples/linux subtree.
#
# Build from the repo root:
#   docker build -f examples/linux/Dockerfile -t speech-core-linux .
#
# Run the integration tests against real model files (mount as /models):
#   scripts/download_models.sh
#   docker run --rm -v "$(pwd)/scripts/models:/models" \
#       speech-core-linux \
#       ctest --test-dir /work/build --output-on-failure
#
# The image is intended for local repro of the Linux CI path (libspeech.so,
# ALSA linkage, ORT-linux). aarch64 cross-builds and QNN runtime are out of
# scope here — see toolchain-aarch64.cmake for the cross path.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    SPEECH_MODEL_DIR=/models

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        ca-certificates \
        libasound2-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . /work

# Fetch ONNX Runtime for the host architecture (linux-x64 or linux-aarch64).
RUN ./examples/linux/setup_linux.sh \
 && test -f ort-linux/include/onnxruntime_c_api.h \
 && test -f ort-linux/lib/libonnxruntime.so

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPEECH_CORE_WITH_ONNX=ON \
        -DSPEECH_CORE_BUILD_EXAMPLES=ON \
        -DORT_DIR=/work/ort-linux \
 && cmake --build build --parallel

# Verify the right artefacts were produced.
RUN file build/examples/linux/libspeech.so | grep -q "shared object" \
 && ldd build/examples/linux/speech_demo | grep -q libasound

# Default: run all ctest targets. Integration tests skip cleanly when
# SPEECH_MODEL_DIR is unset / doesn't contain the expected ONNX files.
CMD ["ctest", "--test-dir", "/work/build", "--output-on-failure"]
