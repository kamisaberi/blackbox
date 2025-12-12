/**
 * @file parser_engine.cpp
 * @brief Implementation of the High-Performance Log Parser.
 */

#include "blackbox/parser/parser_engine.h"
#include "blackbox/common/id_generator.h"
#include "blackbox/common/string_utils.h"
#include "blackbox/common/settings.h"
#include "blackbox/common/logger.h"
#include <cstring>

namespace blackbox::parser {

    // =========================================================
    // Constructor
    // =========================================================
    ParserEngine::ParserEngine() {
        const auto& settings = common::Settings::instance();

        LOG_INFO("Initializing Parser Engine...");

        // Load Vocabulary for Tokenizer
        if (!tokenizer_.load_vocabulary(settings.ai().vocab_path)) {
            LOG_ERROR("Failed to load vocabulary. AI accuracy will be degraded.");
        }

        // Load Scaler Parameters
        if (!scaler_.load_parameters(settings.ai().scaler_path)) {
            LOG_ERROR("Failed to load scaler params. AI inputs will not be normalized.");
        }
    }

    // =========================================================
    // Helper: Extract Field (Move cursor)
    // =========================================================
    std::string_view ParserEngine::extract_field(std::string_view& cursor) {
        if (cursor.empty()) return {};

        // Find position of next space
        size_t space_pos = cursor.find(' ');

        if (space_pos == std::string_view::npos) {
            std::string_view field = cursor;
            cursor = {};
            return field;
        }

        std::string_view field = cursor.substr(0, space_pos);
        cursor.remove_prefix(space_pos + 1);
        return field;
    }

    // =========================================================
    // Process (The Hot Path)
    // =========================================================
    ParsedLog ParserEngine::process(const ingest::LogEvent& raw_event) {
        ParsedLog output;

        // 1. Assign Metadata
        output.id = common::IdGenerator::generate_uuid_v4();
        output.timestamp = raw_event.timestamp_ns;

        // 2. Create View over raw buffer
        std::string_view cursor(raw_event.raw_data, raw_event.length);

        // 3. RFC5424 Syslog Parsing Logic (Optimized)
        // Expected: <PRI>VER TS HOST APP PID MSGID SD MSG
        // Fast-path check for Syslog header
        if (!cursor.empty() && cursor[0] == '<') {
            size_t angle_end = cursor.find('>');
            if (angle_end != std::string_view::npos) {
                cursor.remove_prefix(angle_end + 1); // Skip <PRI>

                // Skip Version if present (digit check)
                if (!cursor.empty() && std::isdigit(cursor[0])) {
                     extract_field(cursor);
                }
            }
        }

        // Timestamp (Skip - we use ingestion time for consistency)
        extract_field(cursor);

        // Hostname
        output.host = extract_field(cursor);

        // Service / App Name
        output.service = extract_field(cursor);
        // Remove trailing colon if present (e.g., "sshd:")
        if (!output.service.empty() && output.service.back() == ':') {
            output.service.remove_suffix(1);
        }

        // Skip PID, MSGID, Structured Data (Simplification for MVP)
        // In a real implementation, we would check for '[' and ']' chars.

        // Remaining string is the Message
        // Trim whitespace using StringUtils
        output.message = common::StringUtils::trim(cursor);

        // Fallback: If parsing failed (empty fields), treat whole raw data as message
        if (output.message.empty()) {
            output.message = std::string_view(raw_event.raw_data, raw_event.length);
            output.host = "unknown";
            output.service = "raw";
        }

        // 4. Vectorize (Text -> Integers)
        tokenizer_.encode(output.message, output.embedding_vector);

        // 5. Scale (Integers -> Normalized Floats)
        scaler_.transform(output.embedding_vector);

        return output;
    }

} // namespace blackbox::parser