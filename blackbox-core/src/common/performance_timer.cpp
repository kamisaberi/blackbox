/**
 * @file performance_timer.cpp
 * @brief Implementation of Low-Overhead Profiling.
 */

#include "blackbox/common/performance_timer.h"
#include "blackbox/common/logger.h"
#include <iostream>

namespace blackbox::common {

    // =========================================================
    // Constructor (Start Clock)
    // =========================================================
    PerformanceTimer::PerformanceTimer(std::string_view name, double warn_threshold_ms)
        : name_(name),
          threshold_ms_(warn_threshold_ms)
    {
        start_time_ = std::chrono::steady_clock::now();
    }

    // =========================================================
    // Destructor (Stop Clock & Report)
    // =========================================================
    PerformanceTimer::~PerformanceTimer() {
        auto end_time = std::chrono::steady_clock::now();

        // Calculate duration in Microseconds for precision
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_
        ).count();

        // Convert to Milliseconds
        double duration_ms = duration_us / 1000.0;

        // Check Threshold
        if (duration_ms > threshold_ms_) {
            // Using std::string concatenation here is acceptable as this is the "Slow Path"
            // (We only pay the cost when something is already wrong)
            LOG_WARN("[SLOW] " + name_ + " took " + std::to_string(duration_ms) + "ms " +
                     "(Threshold: " + std::to_string(threshold_ms_) + "ms)");
        }
    }

} // namespace blackbox::common