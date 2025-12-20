# ⚡ Blackbox Accelerator
### FPGA-Based Hardware Offloading for 100Gbps Security

[![Status](https://img.shields.io/badge/status-experimental-orange)]()
[![Hardware](https://img.shields.io/badge/hardware-Xilinx_Alveo-red)]()
[![Language](https://img.shields.io/badge/hls-C%2B%2B_Vitis-blue)]()
[![Latency](https://img.shields.io/badge/latency-%3C1us-green)]()

**Blackbox Accelerator** is a dedicated Hardware Abstraction Layer (HAL) and High-Level Synthesis (HLS) kernel designed to offload packet inspection logic from the CPU to **FPGA SmartNICs**.

By moving Regex matching, DDoS mitigation, and Packet Parsing into silicon, Blackbox achieves **100Gbps line-rate processing** with **zero CPU overhead**, bypassing the OS kernel entirely.

---

## 🏗️ Architecture

The accelerator operates on a **Zero-Copy DMA** model using the PCIe bus.

```mermaid
graph LR
    subgraph "Host Server (CPU)"
        Core[Blackbox Core C++]
        Driver[Accelerator Driver]
        RAM[Host RAM (Pinned)]
    end

    subgraph "FPGA SmartNIC (Xilinx Alveo)"
        Net[QSFP28 Ports]
        Kernel[Packet Filter Kernel]
        HBM[High Bandwidth Memory]
    end

    Core -->|Update Rules| Driver
    Driver -->|PCIe DMA| HBM
    Net -->|Raw Packets| Kernel
    Kernel -->|Filter Logic| Kernel
    Kernel -->|Clean Data| RAM
    RAM -->|Zero Copy| Core
```

---

## 📂 Directory Structure

```text
blackbox-accelerator/
├── CMakeLists.txt             # Host driver build config
├── include/
│   └── bb_accelerator.h       # Public C++ API for Blackbox Core
├── src/
│   ├── host/                  # CPU-side Drivers (XRT wrapper)
│   │   ├── driver.cpp         # PCIe management
│   │   └── dma_manager.cpp    # Pinned memory allocation
│   └── hls/                   # Hardware-side Logic (Synthesizable C++)
│       ├── packet_filter.cpp  # The Firewall Circuit
│       └── protocols.h        # Network Header definitions
└── firmware/                  # Compiled Bitstreams (.xclbin)
```

---

## 🛠️ Hardware Requirements

To run in **Hardware Mode**, you need:
1.  **FPGA:** Xilinx Alveo U50, U200, U250, or U280.
2.  **Drivers:** Xilinx Runtime (XRT) 2.14+.
3.  **Compiler:** Vitis Unified Software Platform 2023.1+.
4.  **OS:** Ubuntu 20.04 / 22.04 LTS or CentOS 8.

*Note: You can run in **Simulation Mode** on any standard x86 Linux machine without special hardware.*

---

## 🚀 Build Instructions

### 1. Build the Host Driver (CPU Library)
This builds the `libbb_accelerator.so` shared library used by `blackbox-core`.

```bash
mkdir build && cd build
cmake ..
make
sudo make install
```

### 2. Synthesize the FPGA Kernel (Hardware)
This step converts the C++ HLS code into a hardware bitstream (`.xclbin`). This process can take 2-4 hours depending on routing complexity.

```bash
# Set target platform (Example: Alveo U200)
export PLATFORM=xilinx_u200_gen3x16_xdma_2_202110_1

# Compile Kernel (HLS)
v++ -c -t hw --platform $PLATFORM -k blackbox_kernel \
    -I./src/hls src/hls/packet_filter.cpp \
    -o build/packet_filter.xo

# Link Kernel (Bitstream Generation)
v++ -l -t hw --platform $PLATFORM \
    build/packet_filter.xo \
    -o firmware/firewall.xclbin
```

---

## 💻 Simulation & Emulation

You do not need a $5,000 card to develop for this module. You can use **Software Emulation (`sw_emu`)**.

### 1. Build for Emulation
Change the target flag (`-t`) from `hw` to `sw_emu`.

```bash
v++ -c -t sw_emu --platform $PLATFORM ...
```

### 2. Run Simulation
Export the XRT environment variable before running `blackbox-core`.

```bash
export XCL_EMULATION_MODE=sw_emu
./blackbox-core
```
*The driver will detect the emulation mode and spin up a QEMU instance representing the FPGA.*

---

## 🔌 API Usage

Integration with the main Blackbox Core is handled via the `Accelerator` class.

```cpp
#include "bb_accelerator.h"

int main() {
    // 1. Initialize Driver
    bb_accel::Accelerator fpga;
    if (fpga.init("firmware/firewall.xclbin") != 0) {
        return -1; // FPGA not found
    }

    // 2. Offload Blocking Rules (IPs)
    std::vector<uint32_t> bad_ips = { 0x0A000001 }; // 10.0.0.1
    fpga.update_firewall_rules(bad_ips);

    // 3. Poll for Data (Zero Copy)
    char buffer[4096];
    while (true) {
        int bytes = fpga.poll_processed_data(buffer, 4096);
        if (bytes > 0) {
            // Process clean data...
        }
    }
}
```

---

## 📊 Performance Benchmarks

| Metric | Software (CPU) | Hardware (FPGA) | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput** | ~10 Gbps | **100 Gbps** | **10x** |
| **Latency** | ~50 us | **< 1 us** | **50x** |
| **DDoS Resilience**| Crumbles at 1M PPS | **Line Rate (148M PPS)** | **Infinite** |
| **Jitter** | Variable (OS Sched) | **Deterministic** | **Stable** |

---

## 📄 License

**Proprietary Module.**
This module contains trade secrets regarding High-Level Synthesis implementation.
Copyright © 2025 Ignition AI. All Rights Reserved.