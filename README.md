# ◼️ Blackbox
### The AI-Native Flight Recorder for Enterprise & IoT Security

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Core](https://img.shields.io/badge/core-C%2B%2B20-00599C)]()
[![Stack](https://img.shields.io/badge/stack-Go%20%7C%20React%20%7C%20PyTorch-orange)]()
[![License](https://img.shields.io/badge/license-Proprietary-red)]()

> **"Truth survives the crash."**

**Blackbox** is a hyper-performant, distributed **XDR (Extended Detection & Response)** platform designed for the modern threat landscape. Unlike legacy SIEMs that passively index logs for later search, Blackbox sits **inline** with network traffic, using embedded AI to detect, block, and heal threats in microseconds.

It unifies **Enterprise IT** (Cloud, SaaS, Servers) with **OT/IoT** (Factory, Embedded), scaling from a Raspberry Pi gateway to a 100,000 EPS data center cluster.

---

## ⚡ Key Capabilities

### 1. Kinetic Defense Engine (C++20)
Built on `Boost.Asio` and `CUDA`, the core engine handles **100,000+ Events Per Second** on commodity hardware. It features zero-copy parsing, lock-free ring buffers, and micro-batched GPU inference.

### 2. Total Visibility (Logs + Network + Cloud)
Blackbox consumes data from everywhere:
*   **Endpoints:** Syslog and Windows Event Logs via the **Sentry** agent.
*   **Network:** Raw **NetFlow/IPFIX** and **PCAP** sniffing for un-agentable devices.
*   **Cloud & SaaS:** **Vacuum** polls AWS, Azure, SQL, and Slack APIs.

### 3. Inline AI Inference (`xInfer`)
Logs are vectorized and scored by custom Autoencoders/Transformers *before* storage.
*   **Anomaly Detection:** Flags Zero-Day attacks based on behavioral deviation.
*   **Active Defense:** Automatically updates `iptables`/firewalls to block attackers instantly.

### 4. Automated Response (SOAR)
The **Relay** engine connects detection to action.
*   **Notification:** Instant alerts via Slack, PagerDuty, or Email.
*   **Remediation:** Can trigger webhooks to isolate hosts or revoke IAM keys.

---

## 🏗️ System Architecture

Blackbox operates as a Monorepo containing the entire ecosystem.

```mermaid
graph LR
    subgraph "The Edge (IoT/Server)"
        S[Sentry Agent] -->|Protobuf/TCP| B(Ingest Gateway)
    end

    subgraph "The Cloud (Vacuum)"
        AWS[AWS/Azure] --> V[Vacuum Aggregator]
        SQL[Database] --> V
        V -->|Normalized JSON| B
    end

    subgraph "Blackbox Core (C++)"
        B -->|Zero-Copy| C{Parser & AI}
        C -->|Score > 0.9| D[Active Defense]
        D -.->|Block IP| FW[Firewall]
    end

    subgraph "Data & Control"
        C -->|Batch Write| DB[(ClickHouse)]
        C -->|Pub/Sub| R[(Redis)]
        R --> SOAR[Relay (SOAR)]
        R --> API[Tower API]
        API -->|WebSocket| UI[React HUD]
    end
```

### 📂 Module Breakdown

| Directory | Codename | Stack | Responsibility |
| :--- | :--- | :--- | :--- |
| **`/blackbox-core`** | **The Engine** | C++ 20 | High-performance ingestion, AI inference, and routing logic. |
| **`/blackbox-tower`** | **The Tower** | Go | Control Plane API. Authentication, Search, and WebSocket broadcasting. |
| **`/blackbox-hud`** | **The HUD** | React/TS | The Analyst Dashboard. Real-time virtualization of log streams. |
| **`/blackbox-vacuum`** | **The Vacuum** | Go | **Universal Connector.** Polls AWS, Azure, SQL, Slack, and SaaS APIs. |
| **`/blackbox-relay`** | **The Relay** | Go | **SOAR Engine.** Handles notifications (Slack/Jira) and automated workflows. |
| **`/blackbox-sentry`** | **The Agent** | C++ | Standard Endpoint Agent for Linux/Windows servers. |
| **`/blackbox-sentry-micro`** | **Micro** | C99 | Ultra-lightweight IoT agent for embedded devices (<100KB). |
| **`/blackbox-sim`** | **The Lab** | Python | Offline R&D. Trains models and exports artifacts (`.plan`, vocab). |
| **`/blackbox-matrix`** | **The Matrix** | Python | **Cyber Range.** Simulates thousands of devices to stress-test the system. |
| **`/blackbox-deploy`** | **The Chassis** | Docker/K8s | Infrastructure orchestration and database schemas. |

---

## 🛠️ Quick Start

### Prerequisites
*   Docker & Docker Compose (v2.0+)
*   Linux (Recommended) or WSL2

### 1. One-Shot Build
We provide a master script to check dependencies, generate AI artifacts, and build all containers.

```bash
./build_all.sh
```

### 2. Launch the Enterprise Stack
Spin up the Core, API, Database, Dashboard, Vacuum, and Relay.

```bash
./launch_enterprise.sh
```

### 3. Access the System
*   **The HUD (Dashboard):** [http://localhost:3000](http://localhost:3000)
*   **The Tower (API):** [http://localhost:8080/health](http://localhost:8080/health)
*   **ClickHouse (DB):** [http://localhost:8123](http://localhost:8123)

---

## 🌐 Simulation (The Matrix)

Don't have 1,000 servers? Simulate them.
The **Matrix** spins up Docker containers acting as IoT devices, Jetsons, and Servers to generate realistic traffic and attacks.

```bash
cd blackbox-matrix
# 1. Enable ARM Emulation (for IoT simulation)
./setup_qemu.sh

# 2. Run a Scenario (e.g., Mirai Botnet Attack)
python orchestrator.py scenarios/02_mirai_botnet.yaml
```
*Watch the Dashboard Velocity Chart spike as the simulated botnet attacks your Core.*

---

## 🧪 Testing

We use a unified test harness to verify C++ logic, Go APIs, and Python math.

```bash
cd blackbox-tests
./run_all.sh
```

---

## 🗺️ Roadmap

- [x] **Phase I: The Kinetic MVP** (Ingestion, AI, Storage, Dashboard)
- [x] **Phase II: IoT Expansion** (Sentry Micro, Matrix Simulator)
- [x] **Phase III: Enterprise Connectors** (Vacuum for Cloud/SQL, Relay for Slack)
- [ ] **Phase IV: Self-Healing** (The "Architect" Module for IaC fixing)
- [ ] **Phase V: Hardware Acceleration** (FPGA Offloading)

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
Unauthorized copying of this file, via any medium, is strictly prohibited.