# =========================================================
# STAGE 1: BUILDER (Heavy)
# =========================================================
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Toolchain & Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-system-dev \
    libboost-thread-dev \
    libcurl4-openssl-dev \
    libhiredis-dev \
    libmaxminddb-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Workdir
WORKDIR /usr/src/blackbox

# 3. Copy CMake Configuration
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/

# 4. Compile
# We use -j$(nproc) to use all CPU cores for faster compilation
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# =========================================================
# STAGE 2: RUNTIME (Lightweight)
# =========================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 1. Install Runtime Libraries
# These are needed for the binary to link against (Boost, Curl, etc.)
RUN apt-get update && apt-get install -y \
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    libcurl4 \
    libhiredis0.14 \
    libmaxminddb0 \
    iptables \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup App Directory
WORKDIR /app

# 3. Copy Binary from Builder
COPY --from=builder /usr/src/blackbox/build/flight-recorder .

# 4. Create Directories for Config/Models
# These will be mounted via volumes in docker-compose
RUN mkdir -p config models logs

# 5. Runtime Configuration
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 6. Entry Point
# We use exec format so signals (SIGTERM) pass correctly to the C++ app
ENTRYPOINT ["./flight-recorder"]