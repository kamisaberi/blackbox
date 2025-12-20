# 🎖️ Blackbox Commander
### Enterprise Fleet Management & OTA Orchestration

[![Status](https://img.shields.io/badge/status-beta-blue)]()
[![Language](https://img.shields.io/badge/language-Go-00ADD8)]()
[![Protocol](https://img.shields.io/badge/protocol-gRPC-green)]()

**Blackbox Commander** is the "General" of the Blackbox ecosystem. It is responsible for the command, control, and lifecycle management of thousands of distributed **Sentry Agents** (both Linux/Windows Servers and IoT Micro-Agents).

It solves the "Day 2 Operations" problem: How do you update, reconfigure, or monitor the health of 10,000 devices without SSH access?

---

## ⚡ Key Capabilities

### 1. Fleet Registry & Health
*   **Heartbeat Monitoring:** Agents send keep-alives every 30 seconds. Commander tracks online/offline status in real-time.
*   **Metadata Tracking:** Inventory of OS versions, IP addresses, Hostnames, and Agent Versions across the entire fleet.

### 2. Over-The-Air (OTA) Updates
*   **Binary Management:** Upload new Sentry binaries (`v1.2.0`) to the Commander.
*   **Staged Rollouts:** Push updates to "Canary" groups first (e.g., 10% of IoT devices), then roll out to the rest if stable.
*   **Rollback:** Automatically revert if the agent fails to report back after an update.

### 3. Dynamic Configuration
*   **Policy-as-Code:** Define logging policies (e.g., "Enable Debug Mode", "Block USB Storage") in the UI.
*   **Config Push:** Commander generates specific YAML/JSON configs and pushes them via gRPC to agents instantly.

---

## 🏗️ Architecture

Commander uses **gRPC** for high-performance, bidirectional streaming with agents.

```mermaid
graph TD
    Agent[Sentry Agent] -->|gRPC Heartbeat| Cmd(Commander)
    Cmd -->|Query| DB[(PostgreSQL)]
    Cmd -->|Fetch Binary| S3[S3 / MinIO]
    User[Admin] -->|Upload Update| Cmd
    Cmd -->|Push Config| Agent
```

---

## 🛠️ Build Instructions

### Prerequisites
*   Go 1.21+
*   Protocol Buffers Compiler (`protoc`)
*   PostgreSQL (Metadata)
*   MinIO or S3 (Binary Storage)

### 1. Generate Protobufs
Before building the Go code, you must generate the gRPC code from the definitions.

```bash
# Install Go plugins for protoc
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Generate
make proto
```

### 2. Local Build
```bash
go mod tidy
go build -o commander cmd/commander/main.go
./commander
```

### 3. Docker Build
```bash
docker build -t blackbox-commander .
```

---

## ⚙️ Configuration

Commander is configured via Environment Variables.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `COMMANDER_PORT` | gRPC Listening Port | `50051` |
| `DB_DSN` | PostgreSQL Connection String | `postgres://user:pass@localhost:5432/blackbox?sslmode=disable` |
| `S3_ENDPOINT` | URL for Object Storage | `minio:9000` |
| `S3_BUCKET` | Bucket for storing agent binaries | `agent-updates` |
| `S3_ACCESS_KEY` | AWS/MinIO Access Key | - |
| `S3_SECRET_KEY` | AWS/MinIO Secret Key | - |

---

## 🔌 API Definition (Proto)

The communication contract is defined in `api/proto/v1/agent.proto`.

```protobuf
service AgentService {
  // Agent sends heartbeat, receives commands in response
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
  
  // Agent requests latest config
  rpc GetConfig(ConfigRequest) returns (ConfigResponse);
  
  // Agent checks for firmware updates
  rpc CheckUpdate(UpdateCheckRequest) returns (UpdateResponse);
}
```

---

## 📂 Project Structure

```text
api/
└── proto/             # Protocol Buffer definitions
cmd/
└── commander/         # Entry point
internal/
├── grpc/              # gRPC Server implementation
├── db/                # PostgreSQL repositories
├── ota/               # S3/Blob storage logic for binaries
└── config_gen/        # Logic to generate Agent YAMLs dynamically
```

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
