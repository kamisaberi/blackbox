# it seems you designed blackbbox as standalone app running on single workstatetion to monitor its traffic . i want to extentd it to run on many workstations and some features runs on server to monitor all traffics


This is the natural evolution from **MVP** to **Enterprise Platform**.

Currently, your architecture is **Monolithic**:
*   `blackbox-core` (Server) does everything: Ingest, AI, Storage.

To monitor 1,000 workstations, we need a **Distributed Architecture** (Hub-and-Spoke).
1.  **The Hub (Server):** Runs `blackbox-core` (AI, DB, Dashboard).
2.  **The Spoke (Agent):** A new, lightweight binary running on every laptop/server. We will call this **"Blackbox Sentry"**.

---

### **The Distributed Architecture**

```mermaid
graph LR
    subgraph "Workstation A (CEO Laptop)"
        A1[Logs] --> SentryA[Blackbox Sentry]
    end

    subgraph "Workstation B (Web Server)"
        B1[Logs] --> SentryB[Blackbox Sentry]
    end

    subgraph "HQ Server (The Blackbox)"
        SentryA -->|Encrypted TCP| LoadBalancer
        SentryB -->|Encrypted TCP| LoadBalancer
        LoadBalancer --> Core[Blackbox Core (AI)]
        Core --> DB[(ClickHouse)]
        Core --> API[Tower]
    end
```

---

### **Step 1: Create the New Module `blackbox-sentry`**

This is a **Lightweight Forwarder**. It must use < 50MB RAM and < 1% CPU. It does **not** run AI. It simply collects logs and ships them to the Core.

**Folder Structure:**
```text
blackbox/
└── blackbox-sentry/       # [NEW AGENT]
    ├── CMakeLists.txt
    ├── src/
    │   ├── main.cpp
    │   ├── collector/     # Reads local files (/var/log/syslog, Event Viewer)
    │   ├── shipper/       # TCP Client to send data to Core
    │   └── crypto/        # Encrypts traffic (mTLS)
    └── config/
        └── agent.yaml     # Contains "Server_IP" and "Agent_ID"
```

---

### **Step 2: The Code for `blackbox-sentry`**

Here is the code for the **Shipper** (The component that sends data to your Server).

#### **File: `blackbox-sentry/src/shipper/tcp_client.h`**

```cpp
#ifndef BLACKBOX_SENTRY_TCP_CLIENT_H
#define BLACKBOX_SENTRY_TCP_CLIENT_H

#include <boost/asio.hpp>
#include <string>
#include <queue>
#include <mutex>

namespace blackbox::sentry {

    class TcpClient {
    public:
        TcpClient(boost::asio::io_context& io_context,
                  const std::string& server_ip,
                  uint16_t port,
                  const std::string& agent_id);

        // Async connect
        void connect();

        // Send log line
        void send_log(const std::string& log_line);

    private:
        void do_connect();
        void do_write();

        boost::asio::io_context& io_context_;
        boost::asio::ip::tcp::socket socket_;
        boost::asio::ip::tcp::resolver resolver_;

        std::string server_ip_;
        uint16_t port_;
        std::string agent_id_; // Unique ID (e.g., "LAPTOP-01")

        // Outgoing Queue
        std::queue<std::string> write_msg_queue_;
        std::mutex mutex_;
        bool connected_ = false;
    };

}

#endif
```

#### **File: `blackbox-sentry/src/shipper/tcp_client.cpp`**

```cpp
#include "shipper/tcp_client.h"
#include <iostream>

namespace blackbox::sentry {

    TcpClient::TcpClient(boost::asio::io_context& io, const std::string& ip, uint16_t port, const std::string& id)
        : io_context_(io), socket_(io), resolver_(io), server_ip_(ip), port_(port), agent_id_(id) {}

    void TcpClient::connect() {
        do_connect();
    }

    void TcpClient::do_connect() {
        auto endpoints = resolver_.resolve(server_ip_, std::to_string(port_));

        boost::asio::async_connect(socket_, endpoints,
            [this](boost::system::error_code ec, boost::asio::ip::tcp::endpoint) {
                if (!ec) {
                    std::cout << "[SENTRY] Connected to Core at " << server_ip_ << std::endl;
                    connected_ = true;

                    // Handshake: Send Agent ID immediately
                    std::string handshake = "HELLO AGENT_ID=" + agent_id_ + "\n";
                    send_log(handshake);
                } else {
                    // Retry logic would go here
                    std::cerr << "[SENTRY] Connect failed: " << ec.message() << std::endl;
                    connected_ = false;
                }
            });
    }

    void TcpClient::send_log(const std::string& log_line) {
        if (!connected_) return;

        // In a real agent, we need a queue to handle bursts
        // For simple async writing:
        auto msg = std::make_shared<std::string>(log_line + "\n");

        boost::asio::async_write(socket_, boost::asio::buffer(*msg),
            [msg](boost::system::error_code ec, std::size_t /*length*/) {
                if (ec) {
                    std::cerr << "[SENTRY] Write failed" << std::endl;
                }
            });
    }

}
```

---

### **Step 3: Modify `blackbox-core` to Identify Agents**

Your server currently treats every log as "Raw Text." It needs to know **which machine** sent the log.

**1. Update `TcpServer` in `blackbox-core` to handle Handshakes.**

In `src/ingest/tcp_server.cpp`:

```cpp
// Inside process_buffer loop
if (common::StringUtils::starts_with(msg, "HELLO AGENT_ID=")) {
    // This is a handshake!
    std::string agent_id = msg.substr(15); // Extract ID
    this->session_agent_id_ = agent_id;    // Store in TcpSession class
    LOG_INFO("Agent Connected: " + agent_id);
    return; // Don't process as a log
}

// When pushing to RingBuffer, prepend the Agent ID so the AI knows the source
std::string enriched_log = "[" + this->session_agent_id_ + "] " + std::string(chunk);
ring_buffer_.push(enriched_log.data(), enriched_log.size());
```

---

### **Step 4: Central Management (Fleet Control)**

To manage 1,000 agents, you need to update **`blackbox-tower`** (The API).

**1. Create an Agent Registry Table in ClickHouse**
```sql
CREATE TABLE sentry.agents (
    agent_id String,
    ip String,
    last_seen DateTime,
    status String, -- 'Online', 'Offline'
    version String
) ENGINE = ReplacingMergeTree() ORDER BY agent_id;
```

**2. Update Go API to track Heartbeats**
When `blackbox-core` receives a log from `Agent-X`, it should periodically update the `sentry.agents` table: "Agent-X is Alive."

---

### **Summary of Deployment Changes**

**On the Server (The Core):**
*   Run the full stack (Docker Compose).
*   Expose TCP Port `601` to the office network (VPN or Intranet).

**On the Workstations (The Sentries):**
*   Do **not** run Docker.
*   Compile `blackbox-sentry` as a static binary (`g++ -static`).
*   Install as a `systemd` service (Linux) or Windows Service.
*   Config: `server_ip: 192.168.1.100` (Your Core Server).

### **How to Execute This Extension**

1.  **Create the `blackbox-sentry` folder.**
2.  **Copy `CMakeLists.txt`** from Core but remove CUDA/TensorRT dependencies (keep only Boost).
3.  **Implement `FileWatcher`**: Use `std::ifstream` to tail `/var/log/syslog` and call `tcp_client.send_log()`.
4.  **Deploy**: Put the binary on a second laptop, point it to your main laptop's IP, and watch the Dashboard. You will see traffic from the second laptop appear instantly.