/**
 * @file udp_server.cpp
 * @brief Implementation of the Ingestion Layer.
 */

#include "blackbox/ingest/udp_server.h"
#include "blackbox/common/settings.h"
#include "blackbox/common/logger.h"
#include "blackbox/common/metrics.h"
#include "blackbox/ingest/rate_limiter.h"
#include <iostream>

namespace blackbox::ingest {

    // =========================================================
    // Constructor
    // =========================================================
    UdpServer::UdpServer(boost::asio::io_context& io_context, 
                         RingBuffer<65536>& buffer)
        : socket_(io_context, 
                  udp::endpoint(udp::v4(), 
                  common::Settings::instance().network().udp_port)), // Load Port from Settings
          ring_buffer_(buffer)
    {
        LOG_INFO("UDP Server listening on port: " + 
                 std::to_string(common::Settings::instance().network().udp_port));
        
        // Start the infinite loop
        start_receive();
    }

    // =========================================================
    // Start Receive
    // =========================================================
    void UdpServer::start_receive() {
        socket_.async_receive_from(
            boost::asio::buffer(recv_buffer_), 
            remote_endpoint_,
            [this](const boost::system::error_code& error, std::size_t bytes_transferred) {
                this->handle_receive(error, bytes_transferred);
            }
        );
    }

    // =========================================================
    // Handle Receive (The Hot Path)
    // =========================================================
    void UdpServer::handle_receive(const boost::system::error_code& error,
                                   std::size_t bytes_transferred) {
        
        if (!error && bytes_transferred > 0) {
            
            // 1. METRICS: Count raw packet
            common::Metrics::instance().inc_packets_received(1);

            // 2. SECURITY: Rate Limit Check
            // We must convert address to string (allocating memory), 
            // but RateLimiter requires the IP key.
            // Optimization Note: In v2.0, use a raw uint32_t IP for RateLimiter to avoid string alloc.
            std::string source_ip = remote_endpoint_.address().to_string();

            if (!RateLimiter::instance().should_allow(source_ip)) {
                // DoS Detected -> Drop Packet
                common::Metrics::instance().inc_packets_dropped(1);
                
                // Jump to restart loop immediately
                start_receive();
                return; 
            }

            // 3. STORAGE: Push to Lock-Free Buffer
            // We pass the raw pointer and length. formatting happens in the Parser thread.
            bool success = ring_buffer_.push(recv_buffer_.data(), bytes_transferred);

            if (!success) {
                // Buffer Full -> Drop Packet
                common::Metrics::instance().inc_packets_dropped(1);
                
                // Optional: Warn if this happens too often (handled by Metrics Reporter)
            }

        } else if (error != boost::asio::error::operation_aborted) {
            LOG_ERROR("UDP Receive Error: " + error.message());
        }

        // 4. LOOP: Re-arm listener
        start_receive();
    }

} // namespace blackbox::ingest