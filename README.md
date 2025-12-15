# ◼️ Blackbox
### The AI-Native Flight Recorder for Enterprise & IoT Security

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Core](https://img.shields.io/badge/core-C%2B%2B20-00599C)]()
[![IoT](https://img.shields.io/badge/agent-C99-blue)]()
[![Stack](https://img.shields.io/badge/stack-Go%20%7C%20React%20%7C%20PyTorch-orange)]()
[![License](https://img.shields.io/badge/license-Proprietary-red)]()

> **"Truth survives the crash."**

**Blackbox** is a hyper-performant, distributed SIEM (Security Information and Event Management) engine designed for the modern threat landscape. Unlike legacy systems that index logs for later search, Blackbox sits **inline** with network traffic, using embedded AI to detect and block threats in microseconds.

It unifies Enterprise IT security with OT/IoT monitoring, scaling from a Raspberry Pi factory gateway to a 100,000 EPS data center cluster.

---

## ⚡ Key Capabilities

### 1. Kinetic Defense Engine (C++20)
Built on `Boost.Asio` and `CUDA`, the core engine handles **100,000+ Events Per Second** on commodity hardware. It features zero-copy parsing, lock-free ring buffers, and micro-batched GPU inference.

### 2. The IoT "Sentry" Network
Extends visibility to the edge. **`blackbox-sentry-micro`** is a <50KB C99 agent that runs on routers, sensors, and embedded devices, communicating via highly efficient Protocol Buffers to the Core.

### 3. Inline AI Inference
Integrates our proprietary **`xInfer`** technology. Logs are vectorized and scored by custom Autoencoders/Transformers *before* storage.
*   **Anomaly Detection:** Flags Zero-Day attacks based on behavioral deviation.
*   **Active Defense:** Automatically updates `iptables`/firewalls to block attackers instantly.

### 4. Air-Gap Sovereign
Designed for defense and critical infrastructure. Zero dependency on public clouds or external APIs. All models are trained in the local `Sim` lab and deployed as frozen binaries.

---

## 🏗️ System Architecture

Blackbox operates as a Monorepo containing the entire ecosystem.

```mermaid
graph LR
    subgraph "The Edge (IoT/Server)"
        S[Sentry Agent] -->|Protobuf/TCP| B(Ingest Gateway)
    end

    subgraph "Blackbox Core (C++)"
        B -->|Zero-Copy| C{Parser & AI}
        C -->|Score > 0.9| D[Active Defense]
        D -.->|Block IP| FW[Firewall]
    end

    subgraph "Data & Control"
        C -->|Batch Write| DB[(ClickHouse)]
        C -->|Pub/Sub| R[(Redis)]
        R --> API[Tower API (Go)]
        API -->|WebSocket| UI[React HUD]
    end
```

### 📂 Module Breakdown

| Directory | Codename | Stack | Responsibility |
| :--- | :--- | :--- | :--- |
| **`/blackbox-core`** | **The Engine** | C++ 20 | High-performance ingestion, AI inference, and routing logic. |
| **`/blackbox-tower`** | **The Tower** | Go (Golang) | Control Plane API. Authentication, Search, and WebSocket broadcasting. |
| **`/blackbox-hud`** | **The HUD** | React/TS | The Analyst Dashboard. Real-time virtualization of log streams. |
| **`/blackbox-sim`** | **The Lab** | Python | Offline R&D. Trains models and exports artifacts (`.plan`, vocab) for the Core. |
| **`/blackbox-sentry-micro`** | **The Sentry** | C99 | Ultra-lightweight IoT agent. Cross-compilable for ARM/MIPS. |
| **`/blackbox-deploy`** | **The Chassis** | Docker/K8s | Infrastructure orchestration and database schemas. |
| **`/blackbox-tests`** | **QA** | GTest/GoTest | Automated testing suite for all modules. |

---

## 🛠️ Quick Start

### Prerequisites
*   Docker & Docker Compose (v2.0+)
*   Make

### 1. Initialize Data & Brain
Before the system can run, we must generate the AI artifacts (Vocab/Scaler) using the Simulator.

```bash
# 1. Create local data folders
mkdir -p data/config data/models

# 2. Run the AI Simulation to generate config files
cd blackbox-sim
docker build -t blackbox-sim .
docker run --rm -v $(pwd)/../data/config:/app/data/artifacts blackbox-sim
# Output: vocab.txt, scaler_params.txt, model.onnx created in data/config/
```

### 2. Launch the Stack
Spin up the Core, API, Database, and Dashboard.

```bash
cd ../blackbox-deploy
make build
make up
```

### 3. Access the System
*   **The HUD (Dashboard):** [http://localhost:3000](http://localhost:3000)
*   **The Tower (API):** [http://localhost:8080/health](http://localhost:8080/health)
*   **ClickHouse (DB):** [http://localhost:8123](http://localhost:8123)

### 4. Connect an IoT Agent
Simulate a remote device sending telemetry.

```bash
cd ../blackbox-sentry-micro
mkdir build && cd build
cmake .. && make
./sentry-micro
```

---

## 🧪 Testing

We use a unified test harness to verify C++ logic, Go APIs, and Python math.

```bash
cd blackbox-tests
./run_all.sh
```
*   **Core Tests:** Validates RingBuffer memory safety and Token Bucket logic.
*   **Sim Tests:** Verifies Neural Network dimensions and normalization math.
*   **Tower Tests:** Checks API endpoints and Config loading.

---

## 🗺️ Roadmap

- [x] **Phase I: The Kinetic MVP** (Ingestion, AI, Storage, Dashboard)
- [x] **Phase II: IoT Expansion** (Sentry Micro Agent, Protobuf Support)
- [ ] **Phase III: Enterprise Scale** (Kubernetes Helm Charts, RBAC, SSO)
- [ ] **Phase IV: Hardware Acceleration** (FPGA Offloading for 1M+ EPS)

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
Unauthorized copying of this file, via any medium, is strictly prohibited.