/**
 * @file storage_engine.cpp
 * @brief Implementation of Asynchronous Batching.
 */

#include "blackbox/storage/storage_engine.h"
#include <iostream>
#include <chrono>

namespace blackbox::storage {

    // =========================================================
    // Constructor
    // =========================================================
    StorageEngine::StorageEngine() : running_(true) {
        // Start the background flusher immediately
        worker_thread_ = std::thread(&StorageEngine::flush_worker, this);
        std::cout << "[CORE] Storage Engine started. Batch size: " << BATCH_SIZE_THRESHOLD << std::endl;
    }

    // =========================================================
    // Destructor
    // =========================================================
    StorageEngine::~StorageEngine() {
        running_ = false;
        cv_.notify_all(); // Wake up worker to finish
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    // =========================================================
    // Enqueue (Called by AI Thread)
    // =========================================================
    void StorageEngine::enqueue(const parser::ParsedLog& log, float score) {
        
        // 1. BUSINESS LOGIC: FILTERING
        // If score is very low (Safe), we might drop it to save money.
        // For MVP, we save everything > 0.1, drop the rest.
        bool is_alert = (score > 0.8f);
        
        if (score < 0.1f) {
            return; // Drop "Green" noise
        }

        // 2. CONVERT TO DB ROW
        // We need to copy string_views to strings because the raw ringbuffer 
        // memory might be overwritten before the DB write happens.
        DBRow row;
        row.timestamp = log.timestamp;
        row.service = std::string(log.service);
        row.message = std::string(log.message);
        row.anomaly_score = score;
        row.is_alert = is_alert;

        // 3. THREAD-SAFE PUSH
        {
            std::unique_lock<std::mutex> lock(batch_mutex_);
            current_batch_.push_back(std::move(row));
            
            // Optimization: Only notify if we hit the limit
            if (current_batch_.size() >= BATCH_SIZE_THRESHOLD) {
                cv_.notify_one();
            }
        }
    }

    // =========================================================
    // Flush Worker (Background Thread)
    // =========================================================
    void StorageEngine::flush_worker() {
        std::vector<DBRow> outgoing_buffer;
        
        // Reserve memory to avoid reallocations
        outgoing_buffer.reserve(BATCH_SIZE_THRESHOLD);

        while (running_) {
            {
                // Wait for Data OR Timeout
                std::unique_lock<std::mutex> lock(batch_mutex_);
                cv_.wait_for(lock, FLUSH_INTERVAL_MS, [this] { 
                    return !running_ || current_batch_.size() >= BATCH_SIZE_THRESHOLD; 
                });

                // SWAP buffers (Double Buffering)
                // We swap the full 'current_batch' into our local 'outgoing_buffer'
                // and give 'current_batch' an empty vector.
                // This minimizes the time we hold the lock to nanoseconds.
                outgoing_buffer.swap(current_batch_);
            }

            // Lock is released here. AI thread can continue filling 'current_batch'.

            // 4. NETWORK IO
            if (!outgoing_buffer.empty()) {
                send_to_clickhouse(outgoing_buffer);
                outgoing_buffer.clear(); // Ready for next swap
            }
        }
    }

    // =========================================================
    // Send to ClickHouse (Mock Implementation)
    // =========================================================
    void StorageEngine::send_to_clickhouse(const std::vector<DBRow>& batch) {
        // In production, this uses libcurl to POST INSERT to ClickHouse
        
        // Example SQL Construction:
        // INSERT INTO sentry.logs VALUES (time, service, msg, score, is_threat) ...
        
        // std::cout << "[STORAGE] Flushed " << batch.size() << " logs to DB." << std::endl;
        total_written_ += batch.size();
        
        // Simulating IO latency
        // std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

} // namespace blackbox::storage