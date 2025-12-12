/**
 * @file string_utils.h
 * @brief High-performance String Manipulation Utilities.
 *
 * Centralizes common logic for trimming, splitting, and escaping strings.
 * Optimized for std::string_view to minimize heap allocations.
 */

#ifndef BLACKBOX_COMMON_STRING_UTILS_H
#define BLACKBOX_COMMON_STRING_UTILS_H

#include <string>
#include <string_view>
#include <vector>

namespace blackbox::common {

    class StringUtils {
    public:
        /**
         * @brief Removes leading and trailing whitespace.
         * @param str The input view
         * @return A narrowed view of the string (Zero Copy)
         */
        static std::string_view trim(std::string_view str);

        /**
         * @brief Splits a string by a delimiter.
         * @param str The input
         * @param delimiter The char to split by (e.g., ' ')
         * @return Vector of views
         */
        static std::vector<std::string_view> split(std::string_view str, char delimiter);

        /**
         * @brief Escapes special characters for SQL queries.
         * Prevents SQL Injection when writing to ClickHouse.
         *
         * @param str Input string
         * @return Escaped string (e.g., "O'Neil" -> "O\'Neil")
         */
        static std::string escape_sql(const std::string& str);

        /**
         * @brief Checks if a string starts with a prefix.
         */
        static bool starts_with(std::string_view str, std::string_view prefix);
    };

} // namespace blackbox::common

#endif // BLACKBOX_COMMON_STRING_UTILS_H