/**
 * @file tcp_server.h
 * @brief Reliable Log Ingestion (RFC 5424 / RFC 3164 via TCP).
 *
 * Handles multiple concurrent TCP connections.
 * buffers incoming streams and splits them by '\n' delimiters.
 */

#ifndef BLACKBOX_INGEST_TCP_SERVER_H
#define BLACKBOX_INGEST_TCP_SERVER_H

#include <boost/asio.hpp>
#include <memory>
#include <vector>
#include "blackbox/ingest/ring_buffer.h"

namespace blackbox::ingest {

    using boost::asio::ip::tcp;

    // Forward declaration to keep header clean
    class TcpSession;

    class TcpServer {
    public:
        /**
         * @brief Initialize TCP Listener.
         * @param io_context Boost.Asio context
         * @param port Port to listen on (e.g., 601 or 514)
         * @param buffer Reference to shared RingBuffer
         */
        TcpServer(boost::asio::io_context& io_context,
                  uint16_t port,
                  RingBuffer<65536>& buffer);

        ~TcpServer();

    private:
        /**
         * @brief Async accept loop.
         */
        void start_accept();

        boost::asio::io_context& io_context_;
        tcp::acceptor acceptor_;
        RingBuffer<65536>& ring_buffer_;
    };

    /**
     * @brief Represents a single client connection.
     * Keeps a shared_ptr to itself (enable_shared_from_this) to stay alive
     * during async operations.
     */
    class TcpSession : public std::enable_shared_from_this<TcpSession> {
    public:
        TcpSession(tcp::socket socket, RingBuffer<65536>& buffer);

        void start();

    private:
        void do_read();

        /**
         * @brief Scans buffer for newlines and pushes complete logs to RingBuffer.
         */
        void process_buffer(size_t bytes_transferred);

        tcp::socket socket_;
        RingBuffer<65536>& ring_buffer_;

        // 64KB Read Buffer
        enum { max_length = 65536 };
        char data_[max_length];

        // Handling split messages (half a log arriving in one packet, half in next)
        std::string sticky_buffer_;
    };

} // namespace blackbox::ingest

#endif // BLACKBOX_INGEST_TCP_SERVER_H