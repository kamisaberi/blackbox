/**
 * @file logger.h
 * @brief Thread-safe, Color-coded Application Logger.
 */

#ifndef BLACKBOX_COMMON_LOGGER_H
#define BLACKBOX_COMMON_LOGGER_H

#include <string>
#include <string_view>
#include <mutex>

namespace blackbox::common {

    enum class LogLevel {
        DEBUG,
        INFO,
        WARN,
        ERROR,
        CRITICAL
    };

    class Logger {
    public:
        // Delete copy constructors (Singleton-ish usage)
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        static Logger& instance();

        /**
         * @brief Set the minimum log level (e.g., ignore DEBUG in production)
         */
        void set_level(LogLevel level);

        /**
         * @brief Write a message to stdout/stderr safely.
         */
        void log(LogLevel level, std::string_view message, const char* file, int line);

    private:
        Logger() = default;
        
        std::mutex mutex_; // Prevents threads from mixing output
        LogLevel min_level_ = LogLevel::INFO;
    };

} // namespace blackbox::common

// CONVENIENCE MACROS
// Using macros allows us to automatically capture __FILE__ and __LINE__
#define LOG_DEBUG(msg) blackbox::common::Logger::instance().log(blackbox::common::LogLevel::DEBUG, msg, __FILE__, __LINE__)
#define LOG_INFO(msg)  blackbox::common::Logger::instance().log(blackbox::common::LogLevel::INFO, msg, __FILE__, __LINE__)
#define LOG_WARN(msg)  blackbox::common::Logger::instance().log(blackbox::common::LogLevel::WARN, msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) blackbox::common::Logger::instance().log(blackbox::common::LogLevel::ERROR, msg, __FILE__, __LINE__)
#define LOG_CRITICAL(msg) blackbox::common::Logger::instance().log(blackbox::common::LogLevel::CRITICAL, msg, __FILE__, __LINE__)

#endif // BLACKBOX_COMMON_LOGGER_H