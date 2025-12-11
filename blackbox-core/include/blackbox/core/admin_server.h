/**
 * @file admin_server.h
 * @brief Lightweight HTTP Server for Ops.
 * 
 * Exposes /health and /metrics endpoints for Kubernetes and Prometheus.
 * Uses Boost.Asio for async TCP handling.
 */

#ifndef BLACKBOX_CORE_ADMIN_SERVER_H
#define BLACKBOX_CORE_ADMIN_SERVER_H

#include <boost/asio.hpp>
#include <memory>
#include <string>
#include <thread>

namespace blackbox::core {

    using boost::asio::ip::tcp;

    class AdminServer {
    public:
        /**
         * @brief Initialize the Admin Server.
         * 
         * @param port The TCP port to listen on (default 8081)
         */
        explicit AdminServer(short port);
        ~AdminServer();

        /**
         * @brief Starts the server in a background thread.
         */
        void start();

        /**
         * @brief Stops the server.
         */
        void stop();

    private:
        /**
         * @brief The main event loop for the background thread.
         */
        void run_worker();

        /**
         * @brief Async accept new connections.
         */
        void start_accept();

        /**
         * @brief Reads request, generates response, writes back.
         */
        void handle_session(std::shared_ptr<tcp::socket> socket);

        /**
         * @brief Generates the HTTP response string.
         */
        std::string generate_response(const std::string& request_path);

        // Network Resources
        std::shared_ptr<boost::asio::io_context> io_context_;
        std::unique_ptr<tcp::acceptor> acceptor_;
        std::thread worker_thread_;
        
        short port_;
        bool running_ = false;
    };

} // namespace blackbox::core

#endif // BLACKBOX_CORE_ADMIN_SERVER_H