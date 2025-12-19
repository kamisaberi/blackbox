cmake_minimum_required(VERSION 3.15)
project(blackbox-accelerator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Source Files (Host Drivers)
set(SOURCES
    src/host/driver.cpp
)

# Include Directories
include_directories(include)

# Find Xilinx XRT (Optional - allows building in simulation mode if missing)
find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_device.h PATHS /opt/xilinx/xrt/include)
find_library(XRT_LIB NAMES xrt_core PATHS /opt/xilinx/xrt/lib)

if(XRT_INCLUDE_DIR AND XRT_LIB)
    message(STATUS "Xilinx XRT found. Building with Hardware Support.")
    include_directories(${XRT_INCLUDE_DIR})
    link_libraries(${XRT_LIB})
    add_definitions(-DENABLE_HARDWARE)
else()
    message(WARNING "Xilinx XRT not found. Building in Simulation Mode.")
endif()

# Build Shared Library
add_library(bb_accelerator SHARED ${SOURCES})

# Installation
install(TARGETS bb_accelerator DESTINATION lib)
install(FILES include/bb_accelerator.h DESTINATION include)