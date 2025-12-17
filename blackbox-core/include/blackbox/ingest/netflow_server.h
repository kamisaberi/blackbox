/**
 * @file netflow_server.h
 * @brief Ingestion Engine for Network Telemetry.
 * 
 * Listens on UDP 2055.
 * Decodes binary NetFlow packets.
 * Converts them to a standardized Log string for the RingBuffer.
 */

#ifndef BLACKBOX_INGEST_NETFLOW_SERVER_H
#define BLACKBOX_INGEST_NETFLOW_SERVER_H

#include <boost/asio.hpp>
#include <array>
#include <memory>
#include "blackbox/ingest/ring_buffer.h"

namespace blackbox::ingest {

    using boost::asio::ip::udp;

    class NetflowServer {
    public:
        NetflowServer(boost::asio::io_context& io_context, 
                      uint16_t port, 
                      RingBuffer<65536>& buffer);

        ~NetflowServer() = default;

    private:
        void start_receive();
        
        void handle_receive(const boost::system::error_code& error,
                            std::size_t bytes_transferred);

        // Helper to process binary data
        void process_packet(const char* data, size_t length);

        udp::socket socket_;
        udp::endpoint remote_endpoint_;
        RingBuffer<65536>& ring_buffer_;
        
        // NetFlow packets can be up to ~1.5KB, but we alloc more safety
        std::array<char, 4096> recv_buffer_;
    };

} // namespace blackbox::ingest

#endif // BLACKBOX_INGEST_NETFLOW_SERVER_H