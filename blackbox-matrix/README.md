# 🌐 Blackbox Matrix
### Hyper-Scale Cyber Range & Simulation Engine

[![Language](https://img.shields.io/badge/language-Python-blue)]()
[![Platform](https://img.shields.io/badge/platform-Docker%20%7C%20QEMU-2496ED)]()
[![Architecture](https://img.shields.io/badge/arch-x86%20%7C%20ARM32%20%7C%20ARM64-orange)]()

**Blackbox Matrix** is the simulation engine behind the Blackbox platform. It allows developers and security analysts to spin up thousands of virtual devices (IoT sensors, Edge Gateways, Servers) on a single workstation to stress-test the SIEM without needing physical hardware.

It uses **QEMU User Emulation** to run authentic ARM32 and ARM64 binaries inside Docker containers, creating a realistic, multi-architecture "Digital Twin" of an enterprise network.

---

## ⚡ Key Capabilities

### 1. Multi-Architecture Simulation
Matrix doesn't just pretend to be an IoT device; it runs the actual CPU architecture inside Docker.
*   **IoT Nodes:** ARM32 (Raspberry Pi, Thermostats)
*   **Edge AI:** ARM64 (NVIDIA Jetson, AWS Graviton)
*   **Servers:** x86_64 (Linux Databases, Web Servers)

### 2. Scenario-as-Code
Complex attacks are defined in simple YAML files.
*   *Example:* "Spin up 100 normal sensors. After 60 seconds, infect 20 of them with Mirai and start a DDoS attack against the Core."

### 3. Behavior Modeling
Each container runs an **Actor** script driven by a probabilistic behavior profile (JSON).
*   **Weighted Patterns:** "90% chance of 'Sensor OK', 10% chance of 'Connection Error'."
*   **Dynamic Variables:** Automatically injects random IPs, Ports, and Usernames into logs.

---

## 🏗️ Architecture

The Orchestrator parses the scenario and commands the Docker Daemon to spawn the fleet.

```mermaid
graph TD
    User[Analyst] -->|Run Scenario| Orch(Orchestrator)
    Orch -->|Read| YAML[Scenario.yaml]
    Orch -->|Read| JSON[Behavior.json]
    
    subgraph "Docker Host (The Matrix)"
        Orch -->|Spawn| C1[Container: IoT (ARM32)]
        Orch -->|Spawn| C2[Container: Jetson (ARM64)]
        Orch -->|Spawn| C3[Container: Server (x86)]
    end
    
    C1 -->|UDP/514| Core[Blackbox Core]
    C2 -->|UDP/514| Core
    C3 -->|UDP/514| Core
```

---

## 🛠️ Setup & Installation

### Prerequisites
*   Docker Desktop or Docker Engine
*   Python 3.10+
*   Linux (Recommended) or WSL2 on Windows

### 1. Enable Emulation (One-Time Setup)
To run ARM containers on an Intel/AMD machine, you must register the QEMU binfmt handlers.

```bash
./setup_qemu.sh
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build Base Images
Build the simulation container images. This may take a few minutes as it pulls base layers.

```bash
# Uncomment build_images() in orchestrator.py OR run:
docker build -t blackbox-sim-iot ./virtual-devices/iot-arm32
docker build -t blackbox-sim-jetson ./virtual-devices/jetson-arm64
docker build -t blackbox-sim-server ./virtual-devices/server-linux
```

---

## 🚀 Running Simulations

### Basic Usage
Run the orchestrator with a specific scenario file.

```bash
python orchestrator.py scenarios/01_baseline_load.yaml
```

### Available Scenarios

| Scenario File | Description | Target Traffic |
| :--- | :--- | :--- |
| `01_baseline_load.yaml` | Standard healthy network. Web servers and sensors. | ~300 EPS |
| `02_mirai_botnet.yaml` | Simulates an IoT botnet infection and DDoS attempt. | Spikes to 2000+ EPS |
| `03_ransomware.yaml` | Simulates lateral movement and file encryption logs. | Low Volume / High Severity |

---

## 📝 Defining Scenarios

Scenarios are defined in `scenarios/`. Example structure:

```yaml
name: "Mixed Factory Environment"
duration_seconds: 600
core_ip: "172.17.0.1" # Host IP where Blackbox Core listens

nodes:
  - name: "temp-sensor"
    image: "iot-arm32"    # Uses the ARM32 container
    count: 50             # Spawns 50 containers
    behavior: "iot_normal" # Uses behaviors/iot_normal.json
    start_delay: 0

  - name: "rogue-device"
    image: "server-linux"
    count: 1
    behavior: "attack_bruteforce"
    start_delay: 30       # Starts attacking after 30 seconds
```

---

## 📂 Directory Structure

```text
blackbox-matrix/
├── orchestrator.py            # Main control script
├── setup_qemu.sh              # ARM emulation setup
├── scenarios/                 # YAML Mission definitions
├── behaviors/                 # JSON Log patterns
└── virtual-devices/           # Dockerfiles for simulated hardware
    ├── common/actor.py        # The script running inside containers
    ├── iot-arm32/             # Raspberry Pi simulation
    ├── jetson-arm64/          # Edge AI simulation
    └── server-linux/          # Generic Server simulation
```

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
