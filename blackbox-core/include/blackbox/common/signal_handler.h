/**
 * @file signal_handler.h
 * @brief Graceful Shutdown Manager.
 * 
 * Handles SIGINT (Ctrl+C) and SIGTERM (Kubernetes Stop).
 * Allows the application to flush buffers before exiting.
 */

#ifndef BLACKBOX_COMMON_SIGNAL_HANDLER_H
#define BLACKBOX_COMMON_SIGNAL_HANDLER_H

#include <atomic>
#include <mutex>
#include <condition_variable>

namespace blackbox::common {

    class SignalHandler {
    public:
        // Singleton Access
        SignalHandler(const SignalHandler&) = delete;
        SignalHandler& operator=(const SignalHandler&) = delete;

        static SignalHandler& instance();

        /**
         * @brief Registers OS signal interceptors (SIGINT, SIGTERM).
         */
        void register_handlers();

        /**
         * @brief Checks if a shutdown signal has been received.
         * Used by while() loops in worker threads.
         */
        bool is_running() const;

        /**
         * @brief Blocks the main thread until a signal is received.
         * Useful to keep main() alive while threads work.
         */
        void wait_for_signal();

        /**
         * @brief Manually trigger shutdown (e.g., on fatal error).
         */
        void trigger_shutdown();

    private:
        SignalHandler() : running_(true) {}

        static void handle_signal(int signal);

        std::atomic<bool> running_;
        std::mutex mutex_;
        std::condition_variable cv_;
    };

} // namespace blackbox::common

#endif // BLACKBOX_COMMON_SIGNAL_HANDLER_H