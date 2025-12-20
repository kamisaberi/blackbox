# ◼️ Blackbox Core
### High-Performance Kinetic Defense Engine

[![Language](https://img.shields.io/badge/language-C%2B%2B20-00599C)]()
[![Build](https://img.shields.io/badge/build-CMake-brightgreen)]()
[![Performance](https://img.shields.io/badge/performance-100k%20EPS-orange)]()

**Blackbox Core** is the central nervous system of the Blackbox platform. It is a highly optimized C++20 application designed to ingest, parse, analyze, and store security telemetry at **100,000+ Events Per Second (EPS)** on commodity hardware.

Unlike traditional SIEM ingestors (Logstash, Fluentd) that rely on garbage-collected languages, Blackbox Core utilizes a **Push-Based, Zero-Copy** architecture to process data with sub-millisecond latency.

---

## ⚡ Key Architecture

### 1. Ingestion Layer
*   **UDP/TCP Servers:** Built on `Boost.Asio` for asynchronous I/O.
*   **NetFlow v5:** Native binary parsing for network traffic telemetry.
*   **Pcap Sniffer:** Raw packet capture using `libpcap` for deep inspection.

### 2. Memory Management
*   **Ring Buffer:** Uses a lock-free Single-Producer Single-Consumer (SPSC) queue to pass data between the Network Thread and the Processing Thread without mutex contention.
*   **Zero-Copy Parsing:** Extensive use of `std::string_view` to parse logs without heap allocation.

### 3. Intelligence Layer
*   **xInfer Integration:** Bridges C++ to NVIDIA TensorRT to run Deep Learning models (Autoencoders/Transformers) inline.
*   **Rule Engine:** Deterministic signature matching for known threats.
*   **GeoIP:** Memory-mapped IP geolocation using MaxMind DB.

### 4. Active Defense
*   **Alert Manager:** Triggers kinetic responses (e.g., `iptables` blocks) instantly upon threat detection.
*   **Storage Engine:** Asynchronous batch writer for ClickHouse to prevent disk I/O from blocking analysis.

---

## 🛠️ Build Instructions

### Prerequisites
*   **Compiler:** GCC 11+ or Clang 14+ (Must support C++20)
*   **Build System:** CMake 3.20+
*   **Libraries:**
    *   `libboost-all-dev` (1.74+)
    *   `libcurl4-openssl-dev`
    *   `libhiredis-dev`
    *   `libmaxminddb-dev`
    *   `libpcap-dev`
*   **Optional (For AI):** NVIDIA CUDA Toolkit 12.0+ & TensorRT

### Option A: Native Build (Linux)
Use this for development and debugging.

```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure (Release mode for max speed)
cmake -DCMAKE_BUILD_TYPE=Release ..

# 3. Compile (Use all CPU cores)
make -j$(nproc)

# 4. Run
./flight-recorder
```

### Option B: Docker Build (Production)
This creates a minimal production image containing only the compiled binary and runtime libraries.

```bash
# From the blackbox-core/ directory
docker build -f docker/Dockerfile -t blackbox-core .
```

---

## ⚙️ Configuration

Blackbox Core is configured entirely via **Environment Variables**, making it Kubernetes-native.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `BLACKBOX_UDP_PORT` | `514` | Syslog UDP ingestion port. |
| `BLACKBOX_TCP_PORT` | `601` | Reliable Syslog/JSON TCP port. |
| `BLACKBOX_NETFLOW_PORT` | `2055` | NetFlow v5/IPFIX UDP port. |
| `BLACKBOX_ADMIN_PORT` | `8081` | HTTP Port for Prometheus metrics & Health checks. |
| `BLACKBOX_MODEL_PATH` | `models/autoencoder.plan` | Path to the TensorRT AI model. |
| `BLACKBOX_VOCAB_PATH` | `config/vocab.txt` | Path to the tokenizer vocabulary. |
| `BLACKBOX_SCALER_PATH` | `config/scaler_params.txt` | Path to normalization parameters. |
| `BLACKBOX_CLICKHOUSE_URL` | `http://localhost:8123` | ClickHouse HTTP interface. |
| `BLACKBOX_REDIS_HOST` | `localhost` | Redis host for Pub/Sub alerting. |

---

## 🧪 Running Tests

We use **GoogleTest** for unit testing core logic (RingBuffer, RateLimiter, Parser).

```bash
cd build
make unit_tests
./unit_tests
```

---

## 📂 Project Structure

```text
src/
├── ingest/        # Network Listeners (UDP, TCP, NetFlow)
├── parser/        # Log Parsing & Vectorization logic
├── analysis/      # AI Inference & Rule Engine
├── storage/       # Database Batching & Redis Clients
├── enrichment/    # GeoIP & Asset lookup
├── common/        # Utilities (Logger, ThreadPool, Config)
└── core/          # Main Pipeline Orchestrator
```

---

## 📄 License

Copyright © 2025 Ignition AI. All Rights Reserved.
