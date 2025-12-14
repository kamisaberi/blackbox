cmake_minimum_required(VERSION 3.10)
project(blackbox-sentry-micro C)

set(CMAKE_C_STANDARD 99)

# Include Directories
include_directories(
    include
    include/hal
    include/proto
    include/transport
    include/collectors
)

# Source Files
set(SOURCES
    src/main.c
    src/core/agent.c
    
    # HAL (Linux implementation)
    src/hal/linux/time_posix.c
    src/hal/linux/net_posix.c
    src/hal/linux/sys_posix.c

    # Transport
    src/transport/client_tcp.c
    
    # Collectors
    src/collectors/sys_stats.c

    # Proto (Mock/Generated)
    src/proto/pb_wrapper.c
    src/proto/pb_encode.c # The mock we just wrote
    src/proto/telemetry_mock.c # The mock fields
)

# Build
add_executable(sentry-micro ${SOURCES})

# Link Libraries (Linux needs pthread/rt)
if(UNIX)
    target_link_libraries(sentry-micro pthread)
endif()