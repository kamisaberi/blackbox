cmake_minimum_required(VERSION 3.20)
project(blackbox-core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Includes
include_directories(include)

# Find Dependencies (Example)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)

# Define Sources
set(SOURCES
    src/main.cpp
    src/common/logger.cpp
    src/common/settings.cpp
    src/ingest/udp_server.cpp
    src/ingest/ring_buffer.cpp
    src/parser/parser_engine.cpp
    src/parser/tokenizer.cpp
    src/analysis/inference_engine.cpp
    src/analysis/model_loader.cpp
    src/storage/storage_engine.cpp
    src/storage/clickhouse_client.cpp
)

# Build Executable
add_executable(flight-recorder ${SOURCES})

# Link Libraries
target_link_libraries(flight-recorder PRIVATE
    Boost::system
    Threads::Threads
    # Add your xInfer and ClickHouse libs here later
)