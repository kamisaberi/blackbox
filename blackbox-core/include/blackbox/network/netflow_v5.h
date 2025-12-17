/**
 * @file netflow_v5.h
 * @brief Cisco NetFlow v5 Binary Structures.
 * 
 * Reference: Cisco Systems NetFlow Services Export Version 9
 * (v5 is the standard legacy format used by most routers).
 */

#ifndef BLACKBOX_NETWORK_NETFLOW_V5_H
#define BLACKBOX_NETWORK_NETFLOW_V5_H

#include <cstdint>

namespace blackbox::network {

#pragma pack(push, 1) // Disable struct padding

    struct NetflowV5Header {
        uint16_t version;           // NetFlow export version number
        uint16_t count;             // Number of flows exported in this packet (1-30)
        uint32_t sys_uptime;        // Current time in milliseconds since the export device booted
        uint32_t unix_secs;         // Current count of seconds since 0000 UTC 1970
        uint32_t unix_nsecs;        // Residual nanoseconds since 0000 UTC 1970
        uint32_t flow_sequence;     // Sequence counter of total flows seen
        uint8_t  engine_type;       // Type of flow-switching engine
        uint8_t  engine_id;         // Slot number of the flow-switching engine
        uint16_t sampling_interval; // First two bits hold the sampling mode; remaining 14 bits hold value of sampling interval
    };

    struct NetflowV5Record {
        uint32_t src_addr;          // Source IP address
        uint32_t dst_addr;          // Destination IP address
        uint32_t next_hop;          // IP address of next hop router
        uint16_t input_if;          // SNMP index of input interface
        uint16_t output_if;         // SNMP index of output interface
        uint32_t d_pkts;            // Packets in the flow
        uint32_t d_octets;          // Total number of Layer 3 bytes in the packets of the flow
        uint32_t first;             // SysUptime at start of flow
        uint32_t last;              // SysUptime at the time the last packet of the flow was received
        uint16_t src_port;          // TCP/UDP source port number or equivalent
        uint16_t dst_port;          // TCP/UDP destination port number or equivalent
        uint8_t  pad1;              // Unused (zero) bytes
        uint8_t  tcp_flags;         // Cumulative OR of TCP flags
        uint8_t  prot;              // IP protocol type (for example, TCP = 6; UDP = 17)
        uint8_t  tos;               // IP type of service (ToS)
        uint16_t src_as;            // Autonomous system number of the source, either origin or peer
        uint16_t dst_as;            // Autonomous system number of the destination, either origin or peer
        uint8_t  src_mask;          // Source address prefix mask bits
        uint8_t  dst_mask;          // Destination address prefix mask bits
        uint16_t pad2;              // Unused (zero) bytes
    };

#pragma pack(pop) // Restore default alignment

} // namespace blackbox::network

#endif // BLACKBOX_NETWORK_NETFLOW_V5_H