# aarch64 validation environment. Build and run (Docker Desktop or any
# docker with binfmt/QEMU user-mode emulation):
#
#   docker build --platform linux/arm64 -t edgevector-arm64 -f tests/arm64.Dockerfile tests
#   docker run --rm --platform linux/arm64 -v "$PWD:/src:ro" edgevector-arm64 \
#     bash -c "cp -r /src/include /src/tests /tmp/ && cd /tmp/tests && \
#              make clean && make run ARCH=-march=armv8-a"
#
# This validates CORRECTNESS on real AArch64 instruction encodings (the
# Hamming kernel vectorizes to NEON `cnt`); emulated timings are meaningless,
# so performance claims still require real silicon.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ make binutils && rm -rf /var/lib/apt/lists/*
WORKDIR /src/tests
