/**
 * @file udp_server.cpp
 * @brief Implementation of the High-Performance UDP Listener.
 */

#include "blackbox/ingest/udp_server.h"
#include <iostream>

namespace blackbox::ingest {

    // =========================================================
    // Constructor
    // =========================================================
    UdpServer::UdpServer(boost::asio::io_context& io_context, 
                         short port, 
                         RingBuffer<65536>& buffer)
        : socket_(io_context, udp::endpoint(udp::v4(), port)),
          ring_buffer_(buffer),
          dropped_packets_count_(0)
    {
        std::cout << "[CORE] UDP Server listening on port " << port << std::endl;
        
        // Start the infinite loop
        start_receive();
    }

    // =========================================================
    // start_receive
    // =========================================================
    void UdpServer::start_receive() {
        // Boost.Asio async pattern:
        // 1. Give it a buffer to write into (recv_buffer_)
        // 2. Give it a function to call when done (handle_receive)
        socket_.async_receive_from(
            boost::asio::buffer(recv_buffer_), 
            remote_endpoint_,
            [this](const boost::system::error_code& error, std::size_t bytes_transferred) {
                this->handle_receive(error, bytes_transferred);
            }
        );
    }

    // =========================================================
    // handle_receive (The Hot Path)
    // =========================================================
    void UdpServer::handle_receive(const boost::system::error_code& error,
                                   std::size_t bytes_transferred) {
        
        if (!error && bytes_transferred > 0) {
            
            // -----------------------------------------------------
            // CRITICAL SECTION: Push to Lock-Free Buffer
            // -----------------------------------------------------
            // Note: We access recv_buffer_.data() directly.
            // We do NOT create a std::string here (allocation is too slow).
            
            bool success = ring_buffer_.push(recv_buffer_.data(), bytes_transferred);

            if (!success) {
                // Buffer is full (Consumer is too slow).
                // We drop the packet to keep the Ingest Thread alive.
                dropped_packets_count_++;
                
                // Optional: Log every 1000 drops to avoid spamming console
                if (dropped_packets_count_ % 1000 == 0) {
                    std::cerr << "[WARN] RingBuffer Full! Dropped " 
                              << dropped_packets_count_ << " packets." << std::endl;
                }
            }
            // -----------------------------------------------------
        } else {
            if (error != boost::asio::error::operation_aborted) {
                std::cerr << "[ERR] UDP Receive Error: " << error.message() << std::endl;
            }
        }

        // Loop: Re-arm the listener for the next packet immediately
        start_receive();
    }

} // namespace blackbox::ingest