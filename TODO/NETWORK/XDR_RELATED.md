Adding **Network Monitoring** moves your project from a **SIEM** (Security Information & Event Management) to an **XDR** (Extended Detection & Response) platform.

A SIEM listens to what devices *say* (logs).
Network Monitoring watches what devices *do* (traffic).

Here are **4 Major Network Features** you can add, ranging from simple health checks to deep packet inspection.

---

### **1. "Flowmaster" (NetFlow / IPFIX Collector)**
**Category:** Traffic Analysis (High Volume)
*   **The Concept:** Routers and Firewalls don't just send Syslogs; they send **NetFlow** (Cisco) or **IPFIX** (Open Standard). This data summarizes network connections (Src IP, Dst IP, Bytes, Duration) without sending the full packet payload.
*   **Why:** It allows you to visualize bandwidth usage and detect "Data Hoarding" (e.g., a hacker downloading 50GB database) even if there are no logs.
*   **Architecture:**
    *   **Protocol:** UDP Port 2055 (Standard NetFlow).
    *   **Implementation:** A new listener in `blackbox-core` similar to `UdpServer`, but with a binary parser for the NetFlow v5/v9 structure.
*   **AI Use Case:** Train an Autoencoder specifically on Flow Data to detect DDoS attacks instantly.

### **2. "Blackbox Sonar" (Active Health Checks)**
**Category:** Availability Monitoring (Uptime)
*   **The Concept:** Instead of waiting for a log saying "Server Down," actively check it.
*   **Capabilities:**
    *   **ICMP Ping:** Is the host alive?
    *   **TCP Port Check:** Is Nginx actually listening on port 80?
    *   **HTTP Status Check:** Does `GET /health` return 200 OK?
    *   **Latency Map:** Visualize network lag between the Core and the Agents.
*   **Where to build:** This fits perfectly in **`blackbox-vacuum`** (Go), as it is a "Polling" activity.

### **3. "Deep Dive" (DPI - Deep Packet Inspection)**
**Category:** Layer 7 Visibility
*   **The Concept:** A log says "Traffic on Port 443." DPI tells you: "This is **YouTube** traffic," "This is **Tor** traffic," or "This is **BitTorrent**."
*   **How:** It analyzes the packet headers and initial handshake bytes to fingerprint the *application*, not just the port.
*   **Library:** Integrate **`nDPI`** (Open Source C library used by Wireshark/ntop) into `blackbox-core`.
*   **Feature:** "Block all BitTorrent traffic on the Corporate Wi-Fi."

### **4. "ARP Watch" (Layer 2 Defense)**
**Category:** Local Network Security
*   **The Concept:** In IoT and Office networks, hackers use **ARP Spoofing** to pretend to be the Gateway (Man-in-the-Middle). Logs won't catch this.
*   **The Feature:** The **Sentry Agent** listens to raw ARP broadcasts on the local LAN.
*   **Logic:**
    *   Maintain a table: `IP 192.168.1.1` = `MAC AA:BB:CC:DD:EE:FF`.
    *   **Alert:** If `IP 192.168.1.1` suddenly claims to have `MAC 11:22:33:44:55:66` (The Attacker's laptop).
*   **Action:** The Agent can send a "Gratuitous ARP" to "heal" the network or isolate the attacker.

---

### **Implementation Plan: NetFlow Collector (C++ Core)**

This is the most high-value feature for Enterprise. Here is how to add it to `blackbox-core`.

#### **1. File Structure**
```text
blackbox-core/
├── src/
│   ├── network/
│   │   ├── netflow_server.cpp  # Listens on UDP 2055
│   │   └── netflow_parser.cpp  # Decodes binary V5/V9 packets
```

#### **2. NetFlow Struct (Header)**
NetFlow v5 is strictly binary. You map a C struct over the raw buffer.

```cpp
// include/blackbox/network/netflow_structs.h
#pragma once
#include <cstdint>

// NetFlow v5 Header (Big Endian network order)
struct NetFlowV5Header {
    uint16_t version;
    uint16_t count;
    uint32_t sys_uptime;
    uint32_t unix_secs;
    uint32_t unix_nsecs;
    uint32_t flow_sequence;
    uint8_t  engine_type;
    uint8_t  engine_id;
    uint16_t sampling_interval;
};

// NetFlow v5 Record
struct NetFlowV5Record {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t next_hop;
    uint16_t input_if;
    uint16_t output_if;
    uint32_t d_pkts;
    uint32_t d_octets; // Bytes
    uint32_t first;
    uint32_t last;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t  pad1;
    uint8_t  tcp_flags;
    uint8_t  prot;     // Protocol (TCP/UDP)
    uint8_t  tos;
    uint16_t src_as;
    uint16_t dst_as;
    uint8_t  src_mask;
    uint8_t  dst_mask;
    uint16_t pad2;
};
```

#### **3. The Implementation Logic**

In `netflow_server.cpp`:

1.  Reuse your `UdpServer` logic (or inherit from it).
2.  In `handle_receive`:
    *   Cast buffer to `NetFlowV5Header*`.
    *   Swap Endianness (`ntohs`, `ntohl`) because network data is Big Endian.
    *   Loop `header->count` times to read records.
    *   Convert `src_addr` (int) to IP String.
3.  **Format as Log:**
    *   Create a string: `NETFLOW: Src=1.2.3.4 Dst=5.6.7.8 Proto=TCP Bytes=5000`
    *   Push to `RingBuffer`.

Now your AI (Autoencoder) can ingest network traffic stats alongside system logs!

### **Implementation Plan: SNMP Collector (Go Vacuum)**

For "Active Monitoring" (CPU load on a Cisco Router), use **SNMP** inside `blackbox-vacuum`.

**File:** `internal/collectors/network/snmp_poller.go`
**Dependency:** `go get github.com/gosnmp/gosnmp`

```go
package network

import (
	"context"
	"fmt"
	"log"
	"time"

	"blackbox-vacuum/internal/transport"
	"github.com/gosnmp/gosnmp"
)

type SNMPCollector struct {
	TargetIP  string
	Community string // usually "public"
}

func (s *SNMPCollector) Start(ctx context.Context, client *transport.CoreClient) {
	gs := &gosnmp.GoSNMP{
		Target:    s.TargetIP,
		Port:      161,
		Community: s.Community,
		Version:   gosnmp.Version2c,
		Timeout:   time.Duration(2) * time.Second,
	}

	if err := gs.Connect(); err != nil {
		log.Printf("[SNMP] Connect Error: %v", err)
		return
	}
	defer gs.Conn.Close()

	ticker := time.NewTicker(30 * time.Second)
	
	// OIDs for Standard Interfaces
	oids := []string{
		"1.3.6.1.2.1.1.5.0", // SysName
		"1.3.6.1.2.1.1.3.0", // Uptime
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			result, err := gs.Get(oids)
			if err != nil {
				continue
			}

			for _, variable := range result.Variables {
				// Convert to JSON/String
                // Send to Core
			}
		}
	}
}
```

### **Strategic Recommendation**

Add **NetFlow (Flowmaster)** first.
It adds high-speed data ingestion that plays to C++'s strengths and provides data that is critical for security (DDoS detection, Data Exfiltration tracking). It fits perfectly into your existing architecture.