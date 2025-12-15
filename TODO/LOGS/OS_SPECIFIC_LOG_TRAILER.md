You have identified the most critical part of the data pipeline: **The Last Mile**.

The **Server (`blackbox-core`)** is the brain; it sits in the cloud or data center. It does **not** reach into operating systems to pull logs.
The **Agent (`blackbox-sentry`)** is the hands; it sits on the Laptop/Server, reads the OS-specific logs, and pushes them to the Core.

Currently, in your `blackbox-sentry-micro`, we only implemented a basic `sys_stats` collector. To make this a real SIEM, we need to add **OS Log Collectors**.

Here is where that code belongs and how to implement it for **Linux** and **Windows**.

---

### **Location in Project Structure**

We need to expand the **`collectors/`** directory in your Agent.

**Current:**
```text
blackbox-sentry/src/collectors/sys_stats.c  (CPU/RAM)
```

**New Structure:**
```text
blackbox-sentry/src/collectors/
├── linux/
│   ├── syslog_tailer.cpp      # Reads /var/log/syslog, auth.log
│   └── auditd_listener.cpp    # Reads Kernel Audit events (execve, connect)
├── windows/
│   └── event_log_reader.cpp   # Reads Windows Event Viewer API
└── file_watcher.cpp           # Generic "Tail -f" logic
```

---

### **1. Linux: Tailing Files (`syslog_tailer.cpp`)**

On Linux, logs are text files. The agent needs to behave like the `tail -f` command.

**Mechanism:**
1.  Open the file (`/var/log/syslog` or `/var/log/auth.log`).
2.  Seek to the end.
3.  Use `inotify` (Linux API) to wait for new data to be written.
4.  Read the new lines and send them to the Core.

**Code Implementation:**

```cpp
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "transport/tcp_client.h"

class SyslogTailer {
public:
    SyslogTailer(const std::string& filepath, blackbox::sentry::TcpClient& client) 
        : filepath_(filepath), client_(client) {}

    void run() {
        std::ifstream file(filepath_);
        if (!file.is_open()) return;

        // Go to end of file (don't re-send old logs on restart)
        file.seekg(0, std::ios::end);

        std::string line;
        while (running_) {
            // Read all new lines
            while (std::getline(file, line)) {
                // Send to Blackbox Core
                // We prepend a tag so the Parser knows it's Linux Syslog
                client_.send_log("LINUX_SYSLOG: " + line);
            }

            // If EOF, clear flag and wait for new data
            if (file.eof()) {
                file.clear(); 
                // In production, use inotify here to avoid CPU spin
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

private:
    std::string filepath_;
    blackbox::sentry::TcpClient& client_;
    bool running_ = true;
};
```

---

### **2. Windows: Event Viewer (`event_log_reader.cpp`)**

Windows does **not** use text files. It uses a binary database (EVTX). You cannot open these with `ifstream`. You must use the **Windows API**.

**Mechanism:**
1.  Use `EvtSubscribe` (Windows API) to listen for events like "Security", "System", or "Application".
2.  When an event occurs (e.g., EventID 4625 - Failed Login), Windows calls your callback function.
3.  Convert the binary XML to a string and send it.

**Code Implementation:**

```cpp
#ifdef _WIN32
#include <windows.h>
#include <winevt.h> // Windows Event Log API
#pragma comment(lib, "wevtapi.lib")

// This callback runs whenever a new log happens in Windows
DWORD WINAPI SubscriptionCallback(
    EVT_SUBSCRIBE_NOTIFY_ACTION action, 
    PVOID pContext, 
    EVT_HANDLE hEvent
) {
    auto* client = static_cast<blackbox::sentry::TcpClient*>(pContext);

    if (action == EvtSubscribeActionDeliver) {
        DWORD bufferSize = 0;
        DWORD propertyCount = 0;
        
        // 1. Render the Event as XML Text
        EvtRender(NULL, hEvent, EvtRenderEventXml, 0, NULL, &bufferSize, &propertyCount);
        
        std::vector<wchar_t> buffer(bufferSize / 2);
        EvtRender(NULL, hEvent, EvtRenderEventXml, bufferSize, buffer.data(), &bufferSize, &propertyCount);

        // 2. Convert WideChar (Windows) to UTF-8 (Blackbox)
        std::wstring ws(buffer.data());
        std::string xml_log(ws.begin(), ws.end()); // Simplified conversion

        // 3. Send to Core
        client->send_log("WIN_EVENT_LOG: " + xml_log);
    }
    return 0; // Success
}

void StartWindowsCollector(blackbox::sentry::TcpClient& client) {
    // Subscribe to the "Security" channel (Logins, Admin actions)
    EVT_HANDLE hSub = EvtSubscribe(
        NULL, NULL, L"Security", 
        L"*", // XPath Query (Grab everything)
        NULL, 
        &client, // Pass client as context
        (EVT_SUBSCRIBE_CALLBACK)SubscriptionCallback, 
        EvtSubscribeToFutureEvents
    );
}
#endif
```

---

### **3. How this integrates with the Core**

The **Agent** simply ships the raw strings. The **Core** needs to understand them.

You need to update **`blackbox-core/src/parser/parser_engine.cpp`** to handle these new formats.

```cpp
ParsedLog ParserEngine::process(const ingest::LogEvent& raw_event) {
    std::string_view msg(raw_event.raw_data, raw_event.length);

    // 1. Detect Source Type
    if (common::StringUtils::starts_with(msg, "LINUX_SYSLOG:")) {
        // Use Regex/Split to parse standard syslog
        return parse_linux(msg);
    } 
    else if (common::StringUtils::starts_with(msg, "WIN_EVENT_LOG:")) {
        // Use an XML Parser (like TinyXML2) to extract <EventID>
        return parse_windows_xml(msg);
    }
    
    // ... existing logic ...
}
```

### **Summary of Responsibility**

1.  **Blackbox Core:** Does NOT collect. It listens.
2.  **Blackbox Sentry (Linux):** Runs as a daemon (`systemd`). Tails `/var/log/` files.
3.  **Blackbox Sentry (Windows):** Runs as a Service. Hooks into the `wevtapi.dll` API.

To make your project a "Real SIEM," you must compile the Agent (`blackbox-sentry`) separately for Linux and Windows and install it on the target machines.




---




This guide provides the **step-by-step, copy-paste instructions** to build and install the **Blackbox Sentry** agent on production **Linux** servers (Debian/Ubuntu/RHEL) and **Windows** workstations.

We will create a **single C++ codebase** that detects the operating system at compile time and builds the correct collector (Syslog for Linux, Event Viewer for Windows).

---

### **Part 1: The Agent Source Code**

Before building, ensure your `blackbox-sentry` directory has this specific cross-platform structure.

**Directory:** `blackbox/blackbox-sentry/`

#### **1. `CMakeLists.txt` (Cross-Platform Build System)**
This script automatically detects if you are on Linux or Windows and links the correct libraries.

```cmake
cmake_minimum_required(VERSION 3.15)
project(blackbox-sentry VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(Boost REQUIRED COMPONENTS system thread asio)

# Source Files
set(SOURCES 
    src/main.cpp
    src/transport/tcp_client.cpp
)

# OS Specific Configuration
if(WIN32)
    message(STATUS "Building for Windows")
    list(APPEND SOURCES src/collectors/windows/event_log_reader.cpp)
    # Link Windows Event Log API
    link_libraries(wevtapi) 
    add_definitions(-D_WIN32_WINNT=0x0601) # Target Windows 7 or later
else()
    message(STATUS "Building for Linux")
    list(APPEND SOURCES src/collectors/linux/syslog_tailer.cpp)
    # Link PThread for Linux
    find_package(Threads REQUIRED)
endif()

add_executable(blackbox-sentry ${SOURCES})

# Link Libraries
if(WIN32)
    target_link_libraries(blackbox-sentry PRIVATE Boost::system Boost::thread)
else()
    target_link_libraries(blackbox-sentry PRIVATE Boost::system Boost::thread Threads::Threads)
endif()
```

#### **2. `src/main.cpp` (The Entry Point)**
This file switches logic based on the OS.

```cpp
#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include "transport/tcp_client.h"

// Conditional Includes
#ifdef _WIN32
    #include "collectors/windows/event_log_reader.h"
#else
    #include "collectors/linux/syslog_tailer.h"
#endif

// Configuration (In prod, read this from a config.yaml)
const std::string SERVER_IP = "192.168.1.100"; // CHANGE THIS TO YOUR CORE IP
const uint16_t SERVER_PORT = 601;
const std::string AGENT_ID = "SENTRY-NODE-01"; 

int main() {
    try {
        boost::asio::io_context io_context;
        
        // 1. Initialize Network
        blackbox::sentry::TcpClient client(io_context, SERVER_IP, SERVER_PORT, AGENT_ID);
        client.connect();

        // 2. Start Background IO Thread
        std::thread io_thread([&io_context](){ io_context.run(); });

        std::cout << "[SENTRY] Agent Started. Target: " << SERVER_IP << std::endl;

        // 3. Start OS Specific Collector
        #ifdef _WIN32
            std::cout << "[SENTRY] Starting Windows Event Collector..." << std::endl;
            StartWindowsCollector(client); // Blocking call inside internal loop
        #else
            std::cout << "[SENTRY] Starting Linux Syslog Tailer..." << std::endl;
            // Tail auth.log (Debian/Ubuntu) or secure (RHEL/CentOS)
            SyslogTailer tailer("/var/log/auth.log", client);
            tailer.run(); // Blocking call
        #endif

        if(io_thread.joinable()) io_thread.join();

    } catch (std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
    }
    return 0;
}
```

---

### **Part 2: Linux Build & Install (Ubuntu/Debian)**

**Target:** Web Servers, Database Servers.

#### **Step 1: Install Dependencies**
You need the GNU Compiler Collection and Boost.
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev git
```

#### **Step 2: Build the Binary**
```bash
cd blackbox/blackbox-sentry
mkdir build && cd build
cmake ..
make -j$(nproc)
```
*Result:* You now have an executable named `blackbox-sentry`.

#### **Step 3: Installation (Systemd)**
To make it run automatically on boot and restart if it crashes, we create a **Systemd Service**.

1.  **Move Binary:**
    ```bash
    sudo mv blackbox-sentry /usr/local/bin/
    sudo chmod +x /usr/local/bin/blackbox-sentry
    ```

2.  **Create Service File:**
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
    # Root is needed to read /var/log/auth.log. 
    # In strict environments, create a 'sentry' user and use ACLs.

    [Install]
    WantedBy=multi-user.target
    ```

3.  **Enable & Start:**
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable blackbox-sentry
    sudo systemctl start blackbox-sentry
    ```

4.  **Verify:**
    ```bash
    sudo systemctl status blackbox-sentry
    ```

---

### **Part 3: Windows Build & Install**

**Target:** Employee Laptops, Active Directory Controllers.
**Prerequisite:** You need **Visual Studio 2022** (Community Edition is free) with "Desktop development with C++" installed.

#### **Step 1: Install Dependencies (vcpkg)**
Managing C++ libraries on Windows is hard. We use **vcpkg** (Microsoft's package manager).

1.  Open **PowerShell (Admin)**.
2.  Install vcpkg:
    ```powershell
    cd C:\
    git clone https://github.com/microsoft/vcpkg.git
    .\vcpkg\bootstrap-vcpkg.bat
    ```
3.  Install Boost (This takes 10-15 minutes, grab a coffee):
    ```powershell
    .\vcpkg\vcpkg install boost:x64-windows
    ```
4.  Integrate with Visual Studio/CMake:
    ```powershell
    .\vcpkg\vcpkg integrate install
    ```

#### **Step 2: Build the Binary**
1.  Navigate to your source code folder in PowerShell.
2.  Run CMake specifying the toolchain file provided by vcpkg:
    ```powershell
    cd blackbox\blackbox-sentry
    mkdir build
    cd build
    
    # Replace path to vcpkg.cmake with your actual path
    cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
    
    # Compile Release version
    cmake --build . --config Release
    ```
*Result:* You will find `blackbox-sentry.exe` inside `build\Release\`.

#### **Step 3: Installation (Windows Service)**
Running an `.exe` manually is bad (it closes when you log off). We need to register it as a **Windows Service**.

The easiest, most stable way to turn any `.exe` into a service is using **NSSM (Non-Sucking Service Manager)**. It's the industry standard for custom agents.

1.  **Download NSSM:** [https://nssm.cc/download](https://nssm.cc/download)
2.  **Prepare Folder:**
    *   Create `C:\Program Files\Blackbox`
    *   Copy `blackbox-sentry.exe` to that folder.
3.  **Install Service:**
    Open **Command Prompt as Administrator**:
    ```cmd
    nssm.exe install BlackboxSentry "C:\Program Files\Blackbox\blackbox-sentry.exe"
    ```
4.  **Configure Restart Policy (Optional):**
    ```cmd
    nssm.exe set BlackboxSentry AppExit Default Restart
    ```
5.  **Start Service:**
    ```cmd
    sc start BlackboxSentry
    ```

#### **Step 4: Verify**
1.  Open **Services** (`services.msc`).
2.  Look for **BlackboxSentry**. It should be "Running".
3.  Check your Blackbox Dashboard. You should see logs arriving from the Windows machine.

---

### **Summary of Files Needed for Compilation**

Ensure these files exist in `blackbox-sentry/src/collectors/` before building.

**`src/collectors/linux/syslog_tailer.h`**
```cpp
#pragma once
#include <string>
#include "../../transport/tcp_client.h"
class SyslogTailer {
public:
    SyslogTailer(const std::string& path, blackbox::sentry::TcpClient& client);
    void run();
};
```

**`src/collectors/windows/event_log_reader.h`**
```cpp
#pragma once
#include "../../transport/tcp_client.h"
#ifdef _WIN32
void StartWindowsCollector(blackbox::sentry::TcpClient& client);
#endif
```

**(Note: The implementations `.cpp` for these were provided in the previous chat response. Ensure they are saved in the correct folders.)**