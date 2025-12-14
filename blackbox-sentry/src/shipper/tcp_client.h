#ifndef BLACKBOX_SENTRY_TCP_CLIENT_H
#define BLACKBOX_SENTRY_TCP_CLIENT_H

#include <boost/asio.hpp>
#include <string>
#include <queue>
#include <mutex>

namespace blackbox::sentry {

    class TcpClient {
    public:
        TcpClient(boost::asio::io_context& io_context, 
                  const std::string& server_ip, 
                  uint16_t port,
                  const std::string& agent_id);

        // Async connect
        void connect();

        // Send log line
        void send_log(const std::string& log_line);

    private:
        void do_connect();
        void do_write();

        boost::asio::io_context& io_context_;
        boost::asio::ip::tcp::socket socket_;
        boost::asio::ip::tcp::resolver resolver_;
        
        std::string server_ip_;
        uint16_t port_;
        std::string agent_id_; // Unique ID (e.g., "LAPTOP-01")

        // Outgoing Queue
        std::queue<std::string> write_msg_queue_;
        std::mutex mutex_;
        bool connected_ = false;
    };

}

#endif