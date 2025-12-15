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