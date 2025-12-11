/**
 * @file storage_engine.h
 * @brief Thread-safe Batch Writer for ClickHouse.
 * 
 * Manages background threads to flush logs to the database 
 * without blocking the main AI pipeline.
 */

#ifndef BLACKBOX_STORAGE_STORAGE_ENGINE_H
#define BLACKBOX_STORAGE_STORAGE_ENGINE_H

#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include "blackbox/parser/parser_engine.h" // For ParsedLog definition

namespace blackbox::storage {

    // Represents a row to be inserted into ClickHouse
    struct DBRow {
        uint64_t timestamp;
        std::string service;
        std::string message;
        float anomaly_score;
        bool is_alert;
    };

    class StorageEngine {
    public:
        StorageEngine();
        ~StorageEngine();

        /**
         * @brief Adds a processed log to the write queue.
         * 
         * This method is called by the AI Thread. It must be very fast.
         * It just locks a mutex and pushes to a vector.
         * 
         * @param log The data extracted by the Parser
         * @param score The float output from xInfer
         */
        void enqueue(const parser::ParsedLog& log, float score);

    private:
        /**
         * @brief The background loop that sends HTTP requests to ClickHouse.
         */
        void flush_worker();

        /**
         * @brief Sends a batch of rows to the DB.
         * (In a real app, this wraps libcurl or clickhouse-cpp)
         */
        void send_to_clickhouse(const std::vector<DBRow>& batch);

        // CONFIGURATION
        const size_t BATCH_SIZE_THRESHOLD = 1000;
        const std::chrono::milliseconds FLUSH_INTERVAL_MS{1000};

        // STATE
        std::atomic<bool> running_;
        std::vector<DBRow> current_batch_;
        
        // CONCURRENCY
        std::mutex batch_mutex_;
        std::condition_variable cv_;
        std::thread worker_thread_;
        
        // METRICS
        uint64_t total_written_ = 0;
    };

} // namespace blackbox::storage

#endif // BLACKBOX_STORAGE_STORAGE_ENGINE_H