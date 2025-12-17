/**
 * @file netflow_server.cpp
 * @brief Implementation of NetFlow Decoding.
 */

#include "blackbox/ingest/netflow_server.h"
#include "blackbox/network/netflow_v5.h"
#include "blackbox/common/logger.h"
#include "blackbox/common/metrics.h"
#include <iostream>
#include <sstream>
#include <arpa/inet.h> // For ntohl, ntohs (Network to Host conversion)

namespace blackbox::ingest {

    using namespace blackbox::network;

    // =========================================================
    // Constructor
    // =========================================================
    NetflowServer::NetflowServer(boost::asio::io_context& io_context, 
                                 uint16_t port, 
                                 RingBuffer<65536>& buffer)
        : socket_(io_context, udp::endpoint(udp::v4(), port)),
          ring_buffer_(buffer)
    {
        LOG_INFO("NetFlow Server listening on UDP port: " + std::to_string(port));
        start_receive();
    }

    // =========================================================
    // Receive Loop
    // =========================================================
    void NetflowServer::start_receive() {
        socket_.async_receive_from(
            boost::asio::buffer(recv_buffer_),
            remote_endpoint_,
            [this](const boost::system::error_code& error, std::size_t bytes) {
                this->handle_receive(error, bytes);
            }
        );
    }

    void NetflowServer::handle_receive(const boost::system::error_code& error,
                                       std::size_t bytes_transferred) {
        if (!error && bytes_transferred > 0) {
            common::Metrics::instance().inc_packets_received(1);
            process_packet(recv_buffer_.data(), bytes_transferred);
        } else if (error != boost::asio::error::operation_aborted) {
            LOG_ERROR("NetFlow Receive Error: " + error.message());
        }
        start_receive();
    }

    // =========================================================
    // Helpers for IP String Conversion
    // =========================================================
    static std::string ip_to_str(uint32_t ip) {
        char buffer[16];
        // Note: NetFlow IPs are Big Endian. inet_ntop expects Network Byte Order.
        // So we keep it as is (don't swap) if raw struct is Big Endian.
        inet_ntop(AF_INET, &ip, buffer, sizeof(buffer));
        return std::string(buffer);
    }

    // =========================================================
    // Decode Logic
    // =========================================================
    void NetflowServer::process_packet(const char* data, size_t length) {
        // 1. Validate Header Size
        if (length < sizeof(NetflowV5Header)) {
            LOG_WARN("NetFlow packet too short for header");
            return;
        }

        // 2. Cast Header
        const auto* header = reinterpret_cast<const NetflowV5Header*>(data);
        uint16_t version = ntohs(header->version);
        uint16_t count = ntohs(header->count);

        if (version != 5) {
            // For MVP, we only support v5. v9 is much more complex (template based).
            // LOG_DEBUG("Ignored non-v5 NetFlow packet: v" + std::to_string(version));
            return;
        }

        // 3. Loop Records
        const char* current_ptr = data + sizeof(NetflowV5Header);
        size_t expected_size = sizeof(NetflowV5Header) + (count * sizeof(NetflowV5Record));
        
        if (length < expected_size) {
            LOG_WARN("NetFlow packet truncated");
            return;
        }

        for (int i = 0; i < count; ++i) {
            const auto* record = reinterpret_cast<const NetflowV5Record*>(current_ptr);

            // 4. Extract Fields (Convert Endianness)
            uint32_t src_ip = record->src_addr; // Don't swap for inet_ntop
            uint32_t dst_ip = record->dst_addr;
            uint16_t src_port = ntohs(record->src_port);
            uint16_t dst_port = ntohs(record->dst_port);
            uint32_t bytes = ntohl(record->d_octets);
            uint8_t  proto = record->prot;
            uint8_t  tcp_flags = record->tcp_flags;

            // 5. Format as Internal Log Format
            // We create a string representation so the existing ParserEngine can tokenize it.
            // Format: NETFLOW: Src=1.2.3.4:80 Dst=5.6.7.8:443 Proto=6 Bytes=1000 Flags=2
            
            std::stringstream ss;
            ss << "NETFLOW: Src=" << ip_to_str(src_ip) << ":" << src_port
               << " Dst=" << ip_to_str(dst_ip) << ":" << dst_port
               << " Proto=" << (int)proto
               << " Bytes=" << bytes
               << " Flags=" << (int)tcp_flags;

            std::string log_line = ss.str();

            // 6. Push to Ring Buffer
            if (!ring_buffer_.push(log_line.data(), log_line.size())) {
                common::Metrics::instance().inc_packets_dropped(1);
            }

            // Move pointer
            current_ptr += sizeof(NetflowV5Record);
        }
    }

} // namespace blackbox::ingest