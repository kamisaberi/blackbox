/**
 * @file metrics.cpp
 * @brief Implementation of Atomic Stats Tracking.
 */

#include "blackbox/common/metrics.h"
#include "blackbox/common/logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace blackbox::common {

    Metrics& Metrics::instance() {
        static Metrics instance;
        return instance;
    }

    Metrics::~Metrics() {
        stop();
    }

    // =========================================================
    // Atomic Increments (Extremely Fast)
    // =========================================================
    // memory_order_relaxed is sufficient for counters (we don't need synchronization ordering)
    
    void Metrics::inc_packets_received(size_t count) {
        packets_rx_.fetch_add(count, std::memory_order_relaxed);
    }

    void Metrics::inc_packets_dropped(size_t count) {
        packets_dropped_.fetch_add(count, std::memory_order_relaxed);
    }

    void Metrics::inc_inferences_run(size_t count) {
        inferences_.fetch_add(count, std::memory_order_relaxed);
    }

    void Metrics::inc_threats_detected(size_t count) {
        threats_.fetch_add(count, std::memory_order_relaxed);
    }

    void Metrics::inc_db_rows_written(size_t count) {
        db_written_.fetch_add(count, std::memory_order_relaxed);
    }

    void Metrics::inc_db_errors(size_t count) {
        db_errors_.fetch_add(count, std::memory_order_relaxed);
    }

    // =========================================================
    // Reporter Logic
    // =========================================================
    
    void Metrics::start_reporter(int interval_seconds) {
        if (running_) return;
        running_ = true;
        reporter_thread_ = std::thread(&Metrics::reporter_worker, this, interval_seconds);
        LOG_INFO("Metrics Reporter started.");
    }

    void Metrics::stop() {
        if (!running_) return;
        running_ = false;
        if (reporter_thread_.joinable()) {
            reporter_thread_.join();
        }
    }

    void Metrics::reporter_worker(int interval_seconds) {
        
        uint64_t last_rx = 0;
        
        while (running_) {
            // Sleep
            std::this_thread::sleep_for(std::chrono::seconds(interval_seconds));
            if (!running_) break;

            // Snapshot values
            uint64_t rx = packets_rx_.load(std::memory_order_relaxed);
            uint64_t drop = packets_dropped_.load(std::memory_order_relaxed);
            uint64_t infer = inferences_.load(std::memory_order_relaxed);
            uint64_t alerts = threats_.load(std::memory_order_relaxed);
            uint64_t db = db_written_.load(std::memory_order_relaxed);

            // Calculate EPS (Events Per Second) since last report
            uint64_t delta = rx - last_rx;
            double eps = (double)delta / interval_seconds;
            last_rx = rx;

            // Format Report
            std::stringstream ss;
            ss << "STATS [" << interval_seconds << "s] | "
               << "EPS: " << std::fixed << std::setprecision(1) << eps << " | "
               << "Total RX: " << rx << " | "
               << "Drops: " << drop << " | "
               << "Inferred: " << infer << " | "
               << "Threats: " << alerts << " | "
               << "DB Writes: " << db;

            // Log it
            LOG_INFO(ss.str());

            // Warning if dropping too much
            if (drop > 0) {
               // In a real scenario, you might calculate % drop rate
            }
        }
    }

} // namespace blackbox::common