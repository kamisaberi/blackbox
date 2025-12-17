Yes. To make **Blackbox** a truly comprehensive Network Monitoring system (XDR), you are missing one critical capability: **Passive Packet Sniffing (Network Tap)**.

Currently, your system relies on devices *sending* data to you (Syslog, NetFlow). But what if a hacker is on a laptop that *isn't* configured to send logs? You won't see them.

You need a module that listens to the **Raw Network Wire** (Promiscuous Mode) to capture **DNS Queries** and **HTTP Headers** flying by.

Here is the code for a **`PcapSniffer`** module using `libpcap`.

---

### **1. Prerequisites**

You need the PCAP library installed on the Linux server.
```bash
sudo apt-get install libpcap-dev
```

You also need to update `blackbox-core/CMakeLists.txt`:
```cmake
find_library(PCAP_LIB pcap REQUIRED)
target_link_libraries(flight-recorder PRIVATE ${PCAP_LIB})
```

---

### **2. The Header (`include/blackbox/network/pcap_sniffer.h`)**

This class opens a network interface (like `eth0`) and captures packets in real-time.

```cpp
/**
 * @file pcap_sniffer.h
 * @brief Raw Packet Capture Service.
 * 
 * Puts the network interface in promiscuous mode to sniff 
 * DNS queries and HTTP headers directly from the wire.
 */

#ifndef BLACKBOX_NETWORK_PCAP_SNIFFER_H
#define BLACKBOX_NETWORK_PCAP_SNIFFER_H

#include <string>
#include <thread>
#include <atomic>
#include <pcap.h>
#include "blackbox/ingest/ring_buffer.h"

namespace blackbox::network {

    class PcapSniffer {
    public:
        /**
         * @brief Initialize Sniffer.
         * @param interface_name The network card to listen on (e.g., "eth0", "wlan0")
         * @param buffer Reference to the central RingBuffer
         */
        PcapSniffer(const std::string& interface_name, 
                    ingest::RingBuffer<65536>& buffer);
        
        ~PcapSniffer();

        void start();
        void stop();

    private:
        void capture_loop();
        
        // Static callback required by libpcap C API
        static void packet_handler(u_char* user_data, 
                                   const struct pcap_pkthdr* pkthdr, 
                                   const u_char* packet);

        // Parsing Helpers
        void parse_dns(const u_char* payload, int len, const std::string& src_ip);

        std::string interface_;
        ingest::RingBuffer<65536>& ring_buffer_;
        
        pcap_t* handle_ = nullptr;
        std::thread worker_thread_;
        std::atomic<bool> running_{false};
    };

} // namespace blackbox::network

#endif // BLACKBOX_NETWORK_PCAP_SNIFFER_H
```

---

### **3. The Implementation (`src/network/pcap_sniffer.cpp`)**

This implementation focuses on **DNS Extraction**. Finding "bad domains" (like `botnet-c2.xyz`) is the #1 way to catch malware on a network.

```cpp
/**
 * @file pcap_sniffer.cpp
 * @brief Implementation of Raw Packet Processing.
 */

#include "blackbox/network/pcap_sniffer.h"
#include "blackbox/common/logger.h"
#include <iostream>
#include <sstream>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/if_ether.h>
#include <arpa/inet.h>

namespace blackbox::network {

    // =========================================================
    // Constructor
    // =========================================================
    PcapSniffer::PcapSniffer(const std::string& interface_name, 
                             ingest::RingBuffer<65536>& buffer)
        : interface_(interface_name), ring_buffer_(buffer) 
    {
        char errbuf[PCAP_ERRBUF_SIZE];

        // Open live device, snaplen 2048 (enough for headers+DNS), promisc mode, 1000ms timeout
        handle_ = pcap_open_live(interface_.c_str(), 2048, 1, 1000, errbuf);
        
        if (handle_ == nullptr) {
            LOG_ERROR("Could not open device " + interface_ + ": " + errbuf);
        } else {
            // Apply BPF Filter: Only UDP Port 53 (DNS)
            // We filter at the kernel level for performance
            struct bpf_program fp;
            if (pcap_compile(handle_, &fp, "udp port 53", 0, PCAP_NETMASK_UNKNOWN) != -1) {
                pcap_setfilter(handle_, &fp);
                LOG_INFO("Pcap Sniffer ready on " + interface_ + " (Filter: udp port 53)");
            }
        }
    }

    PcapSniffer::~PcapSniffer() {
        stop();
        if (handle_) pcap_close(handle_);
    }

    // =========================================================
    // Lifecycle
    // =========================================================
    void PcapSniffer::start() {
        if (!handle_ || running_) return;
        running_ = true;
        worker_thread_ = std::thread(&PcapSniffer::capture_loop, this);
    }

    void PcapSniffer::stop() {
        if (!running_) return;
        running_ = false;
        if (handle_) pcap_breakloop(handle_);
        if (worker_thread_.joinable()) worker_thread_.join();
    }

    void PcapSniffer::capture_loop() {
        // Start capture loop, passing 'this' as user data
        pcap_loop(handle_, 0, packet_handler, reinterpret_cast<u_char*>(this));
    }

    // =========================================================
    // Packet Handler (Static Callback)
    // =========================================================
    void PcapSniffer::packet_handler(u_char* user_data, 
                                     const struct pcap_pkthdr* pkthdr, 
                                     const u_char* packet) {
        
        auto* self = reinterpret_cast<PcapSniffer*>(user_data);
        
        // 1. Parse Ethernet Header
        struct ether_header* eth_header = (struct ether_header*)packet;
        if (ntohs(eth_header->ether_type) != ETHERTYPE_IP) return; // Ignore non-IPv4

        // 2. Parse IP Header
        // Ethernet header is 14 bytes
        const u_char* ip_ptr = packet + 14; 
        struct ip* ip_header = (struct ip*)ip_ptr;
        
        int ip_header_len = ip_header->ip_hl * 4;
        std::string src_ip = inet_ntoa(ip_header->ip_src);

        // 3. Parse UDP Header
        if (ip_header->ip_p == IPPROTO_UDP) {
            const u_char* udp_ptr = ip_ptr + ip_header_len;
            // UDP Header is 8 bytes
            const u_char* payload = udp_ptr + 8;
            int payload_len = pkthdr->len - 14 - ip_header_len - 8;

            if (payload_len > 12) { // Min DNS header size
                self->parse_dns(payload, payload_len, src_ip);
            }
        }
    }

    // =========================================================
    // DNS Parser Logic
    // =========================================================
    void PcapSniffer::parse_dns(const u_char* payload, int len, const std::string& src_ip) {
        // DNS Header is 12 bytes
        // Query section starts at byte 12
        int pos = 12;
        std::string domain_name = "";
        
        // Simple label parser (e.g., 3www6google3com0)
        while (pos < len) {
            int label_len = payload[pos];
            if (label_len == 0) break; // End of name
            if (label_len > 63) return; // Compression pointer/invalid (skip for MVP)
            
            if (!domain_name.empty()) domain_name += ".";
            
            pos++;
            for (int i = 0; i < label_len && pos < len; i++) {
                char c = (char)payload[pos++];
                if (std::isalnum(c) || c == '-' || c == '_') {
                    domain_name += c;
                }
            }
        }

        if (!domain_name.empty()) {
            // Format as a Log for the Core
            std::stringstream ss;
            ss << "DNS_SNIFF: Src=" << src_ip << " Query=" << domain_name;
            std::string log = ss.str();

            // Push to RingBuffer
            self->ring_buffer_.push(log.data(), log.size());
        }
    }

} // namespace blackbox::network
```

---

### **4. Why is this useful?**

1.  **DGA Detection:** Malware often uses "Domain Generation Algorithms" (e.g., `ag45b.info`, `x99a.net`). Your AI (using the **LogBERT** or **1D-CNN** model we discussed) can easily detect these random strings in the `DNS_SNIFF` logs.
2.  **Shadow IT:** You will see queries for `dropbox.com` or `torproject.org` even if the user didn't install the Sentry Agent on their phone, just because they are on the Wi-Fi.

### **Integration**

Add this to `src/core/pipeline.cpp`.

```cpp
#include "blackbox/network/pcap_sniffer.h"

// Constructor
// Listen on "eth0" or "docker0" depending on your setup
pcap_sniffer_ = std::make_unique<network::PcapSniffer>("eth0", ring_buffer_);

// Start
pcap_sniffer_->start();
```

This makes Blackbox a **passive surveillance** tool, not just a log collector.