Here is the standalone **README.md** for the **`blackbox-sentry`** module.

Place this file at **`blackbox/blackbox-sentry/README.md`**.

***

```markdown
# 🛡️ Blackbox Sentry
### Enterprise Endpoint Agent (Linux & Windows)

[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-blue)]()
[![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C)]()
[![Component](https://img.shields.io/badge/component-EDR-red)]()

**Blackbox Sentry** is the standard endpoint agent for servers, workstations, and cloud instances. Unlike the "Micro" version (which is for embedded IoT), this agent leverages the full power of the host OS to perform deep monitoring and active defense.

It acts as the "Hands" of the Blackbox ecosystem—collecting logs, executing remediation commands, and enforcing security policies locally.

---

## ⚡ Key Capabilities

### 1. Cross-Platform Collection
*   **Linux:** Tails files (`/var/log/syslog`, `auth.log`) using `inotify` for zero-latency ingestion.
*   **Windows:** Hooks directly into the **Windows Event Log API (EvtSubscribe)** to capture Security, System, and Application events in real-time.

### 2. Active Defense (EDR)
*   **Honeyfiles:** Places decoy files (e.g., `passwords.xlsx`) in user directories. If any process tries to encrypt or rename them (Ransomware behavior), the agent kills the process tree instantly.
*   **Process Guard:** Enforces blacklists to prevent dangerous binaries (`nmap`, `nc`, `mimikatz`) from launching.

### 3. Enterprise Reliability
*   **Disk Buffering (WAL):** Implements a Write-Ahead Log. If the network goes down, logs are queued to the local disk. When connectivity returns, data is flushed to the Core. **Zero data loss guarantee.**
*   **Resource Caps:** Hard limits on CPU and RAM usage to ensure the security agent never crashes the production server.

---

## 🛠️ Build Instructions

### Prerequisites
*   CMake 3.15+
*   C++ Compiler (GCC, Clang, or MSVC)
*   Boost Libraries (System, Asio, Thread)

### Linux Build (Ubuntu/Debian/RHEL)
```bash
# 1. Install Dependencies
sudo apt-get install build-essential cmake libboost-all-dev

# 2. Build
mkdir build && cd build
cmake ..
make -j$(nproc)

# 3. Run
sudo ./blackbox-sentry
```

### Windows Build (Visual Studio)
We recommend using `vcpkg` for dependency management.

```powershell
# 1. Install Dependencies via vcpkg
.\vcpkg\vcpkg install boost:x64-windows

# 2. Generate Project
mkdir build; cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# 3. Compile
cmake --build . --config Release
```

---

## 🚀 Installation & Deployment

### Linux (Systemd Service)
Create `/etc/systemd/system/blackbox-sentry.service`:

```ini
[Unit]
Description=Blackbox Sentry Agent
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/blackbox-sentry
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable blackbox-sentry
sudo systemctl start blackbox-sentry
```

### Windows (Service)
Use **NSSM** or the built-in `sc` command to register the executable as a service.

```cmd
# Using SC (Administrator Command Prompt)
sc create BlackboxSentry binPath= "C:\Program Files\Blackbox\blackbox-sentry.exe" start= auto
sc start BlackboxSentry
```

---

## ⚙️ Configuration

Configuration is managed via `config.yaml` or Environment Variables.

| Setting | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| **Server IP** | `SENTRY_SERVER_IP` | `127.0.0.1` | IP of Blackbox Core. |
| **Server Port** | `SENTRY_SERVER_PORT` | `601` | TCP Port of Core. |
| **Agent ID** | `SENTRY_AGENT_ID` | `hostname` | Unique ID for this node. |
| **Log Path** | `SENTRY_LOG_PATH` | `/var/log/syslog` | File to tail (Linux only). |

---

## 📂 Project Structure

```text
src/
├── collectors/
│   ├── linux/
│   │   ├── syslog_tailer.cpp    # inotify file watcher
│   │   └── honey_file.cpp       # Ransomware trap logic
│   ├── windows/
│   │   └── event_log_reader.cpp # Windows API hooks
│   └── process_guard.cpp        # Process blocking logic
├── core/
│   ├── disk_queue.cpp           # Persistent Disk Buffer (WAL)
│   └── agent.cpp                # Main Event Loop
├── transport/
│   └── tcp_client.cpp           # Async TCP Sender
└── main.cpp                     # Entry point & OS detection
```

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
```