/**
 * @file udp_server.h
 * @brief Asynchronous UDP Listener using Boost.Asio.
 * 
 * This class runs on the Main Ingestion Thread.
 * It performs ZERO allocations per packet.
 * It writes directly into the RingBuffer.
 */

#ifndef BLACKBOX_INGEST_UDP_SERVER_H
#define BLACKBOX_INGEST_UDP_SERVER_H

#include <boost/asio.hpp>
#include <array>
#include <memory>
#include "blackbox/ingest/ring_buffer.h" // Assuming RingBuffer definition is here

namespace blackbox::ingest {

    using boost::asio::ip::udp;

    class UdpServer {
    public:
        /**
         * @brief Construct a new Udp Server
         * 
         * @param io_context The Boost.Asio IO context (event loop)
         * @param port The UDP port to listen on (usually 514)
         * @param buffer Reference to the RingBuffer to push data into
         */
        UdpServer(boost::asio::io_context& io_context, 
                  short port, 
                  RingBuffer<65536>& buffer);

        // Delete copy constructors to prevent accidental copying of sockets
        UdpServer(const UdpServer&) = delete;
        UdpServer& operator=(const UdpServer&) = delete;

    private:
        /**
         * @brief Initiates the async receive loop.
         */
        void start_receive();

        /**
         * @brief Callback function when a packet arrives.
         * 
         * @param error Error code (if any)
         * @param bytes_transferred Number of bytes received
         */
        void handle_receive(const boost::system::error_code& error,
                            std::size_t bytes_transferred);

        // MEMBER VARIABLES
        udp::socket socket_;
        udp::endpoint remote_endpoint_;
        
        // The destination queue (Reference, not copy)
        RingBuffer<65536>& ring_buffer_;

        // Reusable scratchpad buffer for incoming data.
        // 64KB is the theoretical max size of a UDP packet.
        std::array<char, 65507> recv_buffer_; 
        
        // Metrics
        uint64_t dropped_packets_count_;
    };

} // namespace blackbox::ingest

#endif // BLACKBOX_INGEST_UDP_SERVER_H