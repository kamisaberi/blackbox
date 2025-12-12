/**
 * @file string_utils.cpp
 * @brief Implementation of String Utilities.
 */

#include "blackbox/common/string_utils.h"
#include <algorithm>
#include <cctype> // for std::isspace

namespace blackbox::common {

    // =========================================================
    // Trim (Zero Copy)
    // =========================================================
    std::string_view StringUtils::trim(std::string_view str) {
        // Trim Left
        while (!str.empty() && std::isspace(static_cast<unsigned char>(str.front()))) {
            str.remove_prefix(1);
        }
        
        // Trim Right
        while (!str.empty() && std::isspace(static_cast<unsigned char>(str.back()))) {
            str.remove_suffix(1);
        }

        return str;
    }

    // =========================================================
    // Split (Zero Copy results)
    // =========================================================
    std::vector<std::string_view> StringUtils::split(std::string_view str, char delimiter) {
        std::vector<std::string_view> tokens;
        size_t start = 0;
        size_t end = str.find(delimiter);

        while (end != std::string_view::npos) {
            tokens.push_back(str.substr(start, end - start));
            start = end + 1;
            end = str.find(delimiter, start);
        }

        // Add last token
        if (start < str.size()) {
            tokens.push_back(str.substr(start));
        }

        return tokens;
    }

    // =========================================================
    // Escape SQL (Critical for Security)
    // =========================================================
    std::string StringUtils::escape_sql(const std::string& str) {
        std::string res;
        res.reserve(str.size() + 4); // Heuristic reservation

        for (char c : str) {
            switch (c) {
                case '\'': res += "\\'"; break;
                case '\\': res += "\\\\"; break;
                case '\b': res += "\\b"; break;
                case '\f': res += "\\f"; break;
                case '\r': res += "\\r"; break;
                case '\n': res += "\\n"; break;
                case '\t': res += "\\t"; break;
                case '\0': res += "\\0"; break; // Null byte injection protection
                default:   res += c; break;
            }
        }
        return res;
    }

    // =========================================================
    // Starts With helper
    // =========================================================
    bool StringUtils::starts_with(std::string_view str, std::string_view prefix) {
        return str.size() >= prefix.size() && 
               str.compare(0, prefix.size(), prefix) == 0;
    }

} // namespace blackbox::common