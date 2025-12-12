/**
 * @file time_utils.cpp
 * @brief Implementation of Fast Time Formatting.
 */

#include "blackbox/common/time_utils.h"
#include <ctime>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <iomanip>
#include <sstream>

namespace blackbox::common {

    // =========================================================
    // Get Current Time (Nano)
    // =========================================================
    uint64_t TimeUtils::now_ns() {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();
    }

    // =========================================================
    // Get Current Time (Milli)
    // =========================================================
    uint64_t TimeUtils::now_ms() {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()
        ).count();
    }

    // =========================================================
    // To ClickHouse SQL Format
    // =========================================================
    std::string TimeUtils::to_clickhouse_format(uint64_t timestamp_ms) {
        // Timestamp is usually ms, but time_t needs seconds
        std::time_t seconds = static_cast<std::time_t>(timestamp_ms / 1000);

        // Thread-safe version of localtime
        struct std::tm tm_buf;
        // gmtime_r is thread-safe on Linux. On Windows use gmtime_s.
        #ifdef _WIN32
            gmtime_s(&tm_buf, &seconds);
        #else
            gmtime_r(&seconds, &tm_buf);
        #endif

        // Pre-allocate buffer to avoid string resizing
        // Format: "YYYY-MM-DD HH:MM:SS" is 19 chars + null terminator
        char buffer[24];

        // Using snprintf is generally faster than std::stringstream
        std::snprintf(buffer, sizeof(buffer),
                      "%04d-%02d-%02d %02d:%02d:%02d",
                      tm_buf.tm_year + 1900,
                      tm_buf.tm_mon + 1,
                      tm_buf.tm_mday,
                      tm_buf.tm_hour,
                      tm_buf.tm_min,
                      tm_buf.tm_sec);

        return std::string(buffer);
    }

    // =========================================================
    // To ISO 8601
    // =========================================================
    std::string TimeUtils::to_iso_8601(uint64_t timestamp_ms) {
        std::time_t seconds = static_cast<std::time_t>(timestamp_ms / 1000);
        int millis = timestamp_ms % 1000;

        struct std::tm tm_buf;
        #ifdef _WIN32
            gmtime_s(&tm_buf, &seconds);
        #else
            gmtime_r(&seconds, &tm_buf);
        #endif

        char buffer[32];
        // "YYYY-MM-DDTHH:MM:SS.mmmZ"
        std::snprintf(buffer, sizeof(buffer),
                      "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                      tm_buf.tm_year + 1900,
                      tm_buf.tm_mon + 1,
                      tm_buf.tm_mday,
                      tm_buf.tm_hour,
                      tm_buf.tm_min,
                      tm_buf.tm_sec,
                      millis);

        return std::string(buffer);
    }

    // =========================================================
    // Parse Syslog
    // =========================================================
    uint64_t TimeUtils::parse_syslog_time(const std::string& date_str) {
        // Input: "Dec 12 10:00:00"
        // Need to guess the year. We assume current year.

        std::tm tm_buf = {};

        // Get current time to find current year
        auto t = std::time(nullptr);
        struct std::tm now_tm;
        #ifdef _WIN32
            localtime_s(&now_tm, &t);
        #else
            localtime_r(&t, &now_tm);
        #endif

        // Parse using stringstream (Simpler than strptime for portability)
        std::stringstream ss(date_str);

        // Map Month Name to Index
        static const std::unordered_map<std::string, int> months = {
            {"Jan", 0}, {"Feb", 1}, {"Mar", 2}, {"Apr", 3}, {"May", 4}, {"Jun", 5},
            {"Jul", 6}, {"Aug", 7}, {"Sep", 8}, {"Oct", 9}, {"Nov", 10}, {"Dec", 11}
        };

        std::string mon_str;
        int day, hour, min, sec;
        char sep; // to eat ':'

        ss >> mon_str >> day >> hour >> sep >> min >> sep >> sec;

        if (months.find(mon_str) != months.end()) {
            tm_buf.tm_mon = months.at(mon_str);
        }

        tm_buf.tm_mday = day;
        tm_buf.tm_hour = hour;
        tm_buf.tm_min = min;
        tm_buf.tm_sec = sec;
        tm_buf.tm_year = now_tm.tm_year; // Assume current year
        tm_buf.tm_isdst = -1; // Let system determine DST

        // Handle edge case: If today is Jan 1st and log is Dec 31st, it was last year.
        time_t result = std::mktime(&tm_buf);
        if (result > t + (86400 * 2)) { // If result is in the future (allow 2 days drift)
            tm_buf.tm_year -= 1;
            result = std::mktime(&tm_buf);
        }

        return static_cast<uint64_t>(result) * 1000; // Return MS
    }

} // namespace blackbox::common