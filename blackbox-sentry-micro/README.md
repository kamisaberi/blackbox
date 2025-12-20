Here is the standalone **README.md** for the **`blackbox-sentry-micro`** module.

Place this file at **`blackbox/blackbox-sentry-micro/README.md`**.

***

```markdown
# 🛡️ Blackbox Sentry Micro
### Ultra-Lightweight IoT & Embedded Agent

[![Language](https://img.shields.io/badge/language-C99-blue)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20RTOS-green)]()
[![Protocol](https://img.shields.io/badge/protocol-Protobuf-orange)]()
[![Size](https://img.shields.io/badge/binary-%3C100KB-brightgreen)]()

**Sentry Micro** is a specialized telemetry agent designed for resource-constrained environments where standard agents (Python/Java/Go) cannot run. It targets **IoT devices, Industrial PLCs, Network Routers, and Edge AI Gateways**.

Written in strict **C99**, it prioritizes stability, zero heap fragmentation, and minimal CPU cycles. It communicates with the Blackbox Core using highly compressed **Protocol Buffers** over TCP/TLS.

---

## ⚡ Key Architecture

### 1. Minimal Footprint
*   **Zero Garbage Collection:** Manual memory management designed to run indefinitely without leaks.
*   **Static Allocation:** Avoids `malloc` in the hot path to prevent heap fragmentation on long-running embedded systems.
*   **Binary Size:** Compiles to <100KB (stripped) on ARMv7.

### 2. Hardware Abstraction Layer (HAL)
The core logic is decoupled from the OS.
*   **`src/hal/linux/`**: Implementation for standard Linux (Raspberry Pi, Jetson, OpenWRT).
*   **Portability:** Can be ported to **FreeRTOS**, **Zephyr**, or Bare Metal by implementing just 3 files (`net.h`, `time.h`, `sys.h`).

### 3. Efficient Protocol
*   **NanoPB:** Uses the NanoPB library to encode telemetry into binary Protocol Buffers.
*   **Bandwidth:** Reduces data usage by ~60% compared to JSON/REST, essential for LTE/Satellite links.

---

## 📂 Directory Structure

```text
blackbox-sentry-micro/
├── CMakeLists.txt             # Build configuration
├── toolchains/                # Cross-compilation setup
├── include/
│   ├── sentry.h               # Public API
│   ├── config.h               # Static configuration
│   ├── hal/                   # Hardware Abstraction Layer (Interfaces)
│   └── proto/                 # Protobuf definitions
├── src/
│   ├── core/                  # Main Event Loop
│   ├── collectors/            # Data Gathering (SysStats, Sensors)
│   ├── transport/             # TCP/TLS Client
│   ├── proto/                 # Generated PB code & wrappers
│   └── hal/
│       └── linux/             # POSIX Implementation
└── libs/                      # Dependencies (NanoPB)
```

---

## 🛠️ Build Instructions

### Prerequisites
*   CMake 3.10+
*   GCC or Clang
*   `gcc-arm-linux-gnueabihf` (Only for cross-compiling)

### 1. Native Build (Host Linux)
Use this for testing on a development machine or x86 server.

```bash
mkdir build && cd build
cmake ..
make
./sentry-micro
```

### 2. Cross-Compile (Raspberry Pi / ARM32)
Use the provided toolchain file to build for target hardware.

```bash
mkdir build-arm && cd build-arm
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.cmake ..
make
# Copy the resulting 'sentry-micro' binary to your IoT device
```

---

## ⚙️ Configuration

For embedded systems, configuration is often compile-time to save parsing overhead.
Edit **`include/config.h`**:

```c
// Target Core Server
#define CONFIG_SERVER_IP   "192.168.1.100"
#define CONFIG_SERVER_PORT 601

// Device Identity (Can be overridden by HAL at runtime)
#define CONFIG_DEVICE_ID   "FACTORY-SENSOR-01"

// Reporting Interval (ms)
#define REPORT_INTERVAL_MS 5000
```

---

## 📦 Protocol Specification

Communication uses the schema defined in `telemetry.proto`:

```protobuf
message Metric {
    uint32 timestamp = 1;
    string key = 2;       // e.g. "cpu_temp"
    float value_num = 3;  // e.g. 45.5
}

message Packet {
    string device_id = 1;
    repeated Metric metrics = 2;
}
```

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
```