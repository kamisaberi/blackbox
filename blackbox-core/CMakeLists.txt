cmake_minimum_required(VERSION 3.20)
project(blackbox-core VERSION 0.1.0 LANGUAGES CXX)

# =========================================================
# 1. Compiler Settings
# =========================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization Flags (O3 for Max Speed, -march=native for SIMD)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter -march=native -O3)

# Export symbols for CrashHandler stack traces
if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
endif()

# =========================================================
# 2. Dependencies
# =========================================================

# A. Boost (Networking)
find_package(Boost REQUIRED COMPONENTS system)

# B. Threads (POSIX)
find_package(Threads REQUIRED)

# C. CUDA / TensorRT (AI)
find_package(CUDAToolkit REQUIRED)
# Note: TensorRT usually requires manual path setup or a custom FindTensorRT.cmake
# We assume standard install or Docker container paths here.
include_directories(/usr/include/x86_64-linux-gnu)
link_directories(/usr/lib/x86_64-linux-gnu)

# D. cURL (ClickHouse HTTP)
find_package(CURL REQUIRED)

# E. Hiredis (Redis)
find_library(HIREDIS_LIB hiredis REQUIRED)

# F. MaxMindDB (GeoIP)
find_library(MAXMINDDB_LIB maxminddb REQUIRED)

# G. ExecInfo (Crash Handler - Alpine only, usually built-in on Ubuntu)
# find_library(EXECINFO_LIB execinfo)

# =========================================================
# 3. Include Directories
# =========================================================
include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})

# =========================================================
# 4. Source Files
# =========================================================
# Explicitly listing files ensures we don't miss any module
set(SOURCES
    # Core
    src/main.cpp
    src/core/application.cpp
    src/core/pipeline.cpp
    src/core/admin_server.cpp

    # Ingest
    src/ingest/udp_server.cpp
    src/ingest/tcp_server.cpp
    src/ingest/rate_limiter.cpp
    src/ingest/ring_buffer.cpp

    # Parser
    src/parser/parser_engine.cpp
    src/parser/tokenizer.cpp
    src/parser/feature_scaler.cpp

    # Analysis
    src/analysis/inference_engine.cpp
    src/analysis/model_loader.cpp
    src/analysis/rule_engine.cpp
    src/analysis/alert_manager.cpp
    src/analysis/block_list_manager.cpp

    # Storage
    src/storage/storage_engine.cpp
    src/storage/clickhouse_client.cpp
    src/storage/redis_client.cpp

    # Enrichment
    src/enrichment/geoip_service.cpp

    # Common / Utils
    src/common/settings.cpp
    src/common/logger.cpp
    src/common/metrics.cpp
    src/common/signal_handler.cpp
    src/common/crash_handler.cpp
    src/common/system_stats.cpp
    src/common/thread_utils.cpp
    src/common/string_utils.cpp
    src/common/time_utils.cpp
    src/common/id_generator.cpp
)

# =========================================================
# 5. Build Executable
# =========================================================
add_executable(flight-recorder ${SOURCES})

# =========================================================
# 6. Link Libraries
# =========================================================
target_link_libraries(flight-recorder PRIVATE
    Boost::system
    Threads::Threads
    CUDA::cudart
    nvinfer          # TensorRT
    nvparsers        # ONNX Parser (optional)
    ${CURL_LIBRARIES}
    ${HIREDIS_LIB}
    ${MAXMINDDB_LIB}
    # ${EXECINFO_LIB} # Uncomment for Alpine Linux
)

message(STATUS "Build Configured. Ready to compile Blackbox Core.")