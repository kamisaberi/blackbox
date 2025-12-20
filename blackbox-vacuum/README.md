Here is the standalone **README.md** for the **`blackbox-vacuum`** module.

Place this file at **`blackbox/blackbox-vacuum/README.md`**.

***

```markdown
# 🌪️ Blackbox Vacuum
### Cloud & SaaS Data Aggregator

[![Language](https://img.shields.io/badge/language-Go-00ADD8)]()
[![Type](https://img.shields.io/badge/type-Poller%20%2F%20Webhook-orange)]()
[![Output](https://img.shields.io/badge/output-TCP%20Stream-green)]()

**Blackbox Vacuum** is the universal integration layer for the Blackbox ecosystem.

While the C++ Core is optimized for high-speed *Push* traffic (Syslog, NetFlow), modern enterprise data often lives in APIs that require *Polling* (AWS CloudTrail, Database Audit Logs) or HTTP Webhooks (SaaS).

Vacuum connects to these external sources, fetches the data, normalizes it into a standard JSON stream, sanitizes it (removing secrets), and pushes it to the Core via TCP.

---

## ⚡ Key Capabilities

### 1. The Universal Connector
Connects to "Pull-based" data sources that cannot send Syslogs directly:
*   **Cloud:** AWS CloudTrail, Azure Monitor, GCP Logging.
*   **Databases:** PostgreSQL (`pg_stat_activity`), MSSQL Audit, MongoDB Profiler, Redis.
*   **Identity:** Okta System Log, Auth0.
*   **SaaS:** Slack History, Office 365 Graph API.

### 2. Webhook Server
Acts as a listener for third-party services that send HTTP POSTs:
*   **GitHub/GitLab:** Audit push events and PRs.
*   **Slack/Teams:** Event subscriptions.
*   **Stripe/Salesforce:** Transactional webhooks.

### 3. Data Sanitization (DLP)
Includes a regex-based scrubber (`pkg/utils/sanitizer.go`) that automatically strips sensitive patterns **before** data leaves the collector:
*   AWS Access Keys (`AKIA...`)
*   Private Keys (`-----BEGIN RSA...`)
*   Passwords in JSON payloads.

---

## 🏗️ Architecture

```mermaid
graph LR
    subgraph "External APIs"
        AWS[AWS CloudTrail]
        SQL[PostgreSQL]
        Okta[Okta Identity]
    end

    subgraph "Blackbox Vacuum"
        Poller[Collector Loop]
        Hook[Webhook Server]
        Scrub[Sanitizer]
        TCP[TCP Client]
    end

    AWS --> Poller
    SQL --> Poller
    Okta --> Poller
    GitHub -->|HTTP POST| Hook

    Poller --> Scrub
    Hook --> Scrub
    Scrub --> TCP
    TCP -->|JSON Stream| Core[Blackbox Core :601]
```

---

## 🛠️ Build & Setup

### Prerequisites
*   Go 1.21+

### 1. Automated Setup
Use the provided script to initialize the module and download all SDK dependencies (AWS, Azure, K8s, etc.).

```bash
# Linux / macOS
chmod +x setup_vacuum.sh
./setup_vacuum.sh

# Windows
.\setup_vacuum.bat
```

### 2. Local Execution
```bash
# Set target Core address
export VACUUM_CORE_HOST=localhost
export VACUUM_CORE_PORT=601

# Enable specific collectors via ENV vars
export AWS_REGION=us-east-1
export MSSQL_CONN_STRING="sqlserver://sa:pass@localhost:1433"

# Run
go run cmd/vacuum/main.go
```

### 3. Docker Build
```bash
docker build -t blackbox-vacuum .
```

---

## ⚙️ Configuration

Vacuum is modular. It only starts collectors for services that have configuration provided via Environment Variables.

### Core Settings
| Variable | Default | Description |
| :--- | :--- | :--- |
| `VACUUM_CORE_HOST` | `blackbox-core` | Hostname of the C++ Engine. |
| `VACUUM_CORE_PORT` | `601` | TCP port for ingestion. |
| `VACUUM_WEBHOOK_PORT` | `9090` | Local port for Webhook Server. |

### Collector Flags
Define these to activate specific integrations:

| Variable | Service | Format |
| :--- | :--- | :--- |
| `AWS_REGION` | AWS | e.g., `us-east-1` (Requires `~/.aws/credentials` or IAM Role) |
| `MSSQL_CONN_STRING` | SQL Server | `sqlserver://user:pass@host:port` |
| `POSTGRES_CONN_STRING`| PostgreSQL | `postgres://user:pass@host:5432/db` |
| `MONGO_URI` | MongoDB | `mongodb://user:pass@host:27017` |
| `REDIS_ADDR` | Redis Mon | `localhost:6379` |
| `SLACK_API_TOKEN` | Slack | `xoxb-...` |
| `OKTA_DOMAIN` | Okta | `org.okta.com` |

---

## 📂 Project Structure

```text
blackbox-vacuum/
├── cmd/
│   └── vacuum/
│       └── main.go            # Entry Point & Registry
├── internal/
│   ├── config/                # Env Loader
│   ├── transport/             # TCP Client (Auto-reconnect)
│   ├── collector/             # Interface definition
│   ├── collectors/            # Logic for specific APIs
│   │   ├── cloud/             # AWS, Azure
│   │   ├── database/          # SQL, Mongo, Redis
│   │   ├── identity/          # Okta
│   │   └── saas/              # Slack, O365
│   └── server/                # Webhook HTTP Handlers
└── pkg/
    └── utils/
        ├── sanitizer.go       # Secret Redaction logic
        └── json_fmt.go        # JSON Minifier
```

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
```