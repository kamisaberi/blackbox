/**
 * @file metrics.h
 * @brief Lock-Free Application Observability.
 * 
 * Tracks system performance (EPS, Drops, Latency) using atomic counters.
 * Includes a background reporter to log system health.
 */

#ifndef BLACKBOX_COMMON_METRICS_H
#define BLACKBOX_COMMON_METRICS_H

#include <atomic>
#include <thread>
#include <atomic>

namespace blackbox::common {

    class Metrics {
    public:
        // Singleton Access
        Metrics(const Metrics&) = delete;
        Metrics& operator=(const Metrics&) = delete;
        static Metrics& instance();

        // ==========================================
        // Hot Path Incrementers (Called by Threads)
        // ==========================================
        
        // Ingestion Layer
        void inc_packets_received(size_t count = 1);
        void inc_packets_dropped(size_t count = 1);

        // AI Layer
        void inc_inferences_run(size_t count = 1);
        void inc_threats_detected(size_t count = 1);

        // Storage Layer
        void inc_db_rows_written(size_t count = 1);
        void inc_db_errors(size_t count = 1);

        // ==========================================
        // Management
        // ==========================================
        
        /**
         * @brief Starts a background thread that logs stats every N seconds.
         */
        void start_reporter(int interval_seconds = 5);

        /**
         * @brief Stops the reporter thread.
         */
        void stop();

    private:
        Metrics() = default;
        ~Metrics();

        void reporter_worker(int interval_seconds);

        // ATOMIC COUNTERS (Lock-Free)
        std::atomic<uint64_t> packets_rx_{0};
        std::atomic<uint64_t> packets_dropped_{0};
        std::atomic<uint64_t> inferences_{0};
        std::atomic<uint64_t> threats_{0};
        std::atomic<uint64_t> db_written_{0};
        std::atomic<uint64_t> db_errors_{0};

        // Reporter State
        std::atomic<bool> running_{false};
        std::thread reporter_thread_;
    };

} // namespace blackbox::common

#endif // BLACKBOX_COMMON_METRICS_H