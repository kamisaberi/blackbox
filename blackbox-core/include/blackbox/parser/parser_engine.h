/**
 * @file parser_engine.h
 * @brief High-Performance Log Extraction & Vectorization.
 * 
 * Uses std::string_view to parse logs without memory allocation.
 */

#ifndef BLACKBOX_PARSER_ENGINE_H
#define BLACKBOX_PARSER_ENGINE_H

#include <vector>
#include <string_view>
#include <array>
#include "blackbox/ingest/ring_buffer.h"

namespace blackbox::parser {

    // The structured output after parsing
    struct ParsedLog {
        uint64_t timestamp;
        std::string_view host;    // Points to raw buffer
        std::string_view service; // Points to raw buffer
        std::string_view message; // Points to raw buffer
        
        // The numerical representation for xInfer
        // Fixed size array for stack allocation speed (e.g., 768 dim BERT or 128 dim Autoencoder)
        std::array<float, 128> embedding_vector; 
    };

    class ParserEngine {
    public:
        ParserEngine();
        ~ParserEngine() = default;

        /**
         * @brief Main processing function.
         * 1. Detects format (RFC5424 Syslog vs JSON).
         * 2. Extracts fields.
         * 3. Tokenizes message into floats.
         * 
         * @param raw_event The event popped from RingBuffer
         * @return ParsedLog The structured data
         */
        ParsedLog process(const ingest::LogEvent& raw_event);

    private:
        /**
         * @brief A fast, heuristic-based tokenizer.
         * In production, this would look up a loaded vocabulary HashMap.
         * For MVP, we use hashing (Hashing Trick).
         */
        void vectorize_text(std::string_view text, std::array<float, 128>& out_vector);

        // Helper to find the next space in a raw buffer
        std::string_view extract_field(std::string_view& cursor);
    };

} // namespace blackbox::parser

#endif // BLACKBOX_PARSER_ENGINE_H