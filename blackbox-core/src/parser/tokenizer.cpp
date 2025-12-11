/**
 * @file tokenizer.cpp
 * @brief Implementation of the Vocabulary Loader.
 */

#include "blackbox/parser/tokenizer.h"
#include "blackbox/common/logger.h" // Use our new Logger
#include <fstream>
#include <sstream>
#include <algorithm>

namespace blackbox::parser {

    // =========================================================
    // Constructor
    // =========================================================
    Tokenizer::Tokenizer() {
        // Reserve some space to avoid rehashes if possible
        vocab_map_.reserve(10000);
    }

    // =========================================================
    // Load Vocabulary
    // =========================================================
    bool Tokenizer::load_vocabulary(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open vocabulary file: " + path);
            return false;
        }

        std::string line;
        int index = 0;
        
        // Read file line by line.
        // The Python script 'blackbox-sim' ensures the line number matches the index.
        while (std::getline(file, line)) {
            // Trim whitespace (rtrim)
            line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), line.end());

            if (!line.empty()) {
                vocab_map_[line] = index;
                index++;
            }
        }

        LOG_INFO("Loaded Vocabulary with " + std::to_string(vocab_map_.size()) + " tokens.");
        return true;
    }

    // =========================================================
    // Encode (The Hot Path)
    // =========================================================
    void Tokenizer::encode(std::string_view text, std::array<float, 128>& out_vector) {
        // 1. Reset Vector (Zero out)
        out_vector.fill(0.0f);

        // 2. Tokenize logic
        // We split by space and lookup in the map.
        
        std::string_view remaining = text;
        int vector_idx = 0;

        while (!remaining.empty() && vector_idx < 128) {
            // Find next space
            size_t space_pos = remaining.find(' ');
            std::string_view token_view;

            if (space_pos == std::string_view::npos) {
                token_view = remaining;
                remaining = {}; // Done
            } else {
                token_view = remaining.substr(0, space_pos);
                remaining.remove_prefix(space_pos + 1);
            }

            // Skip empty tokens (double spaces)
            if (token_view.empty()) continue;

            // 3. Lookup
            // We must convert view to string for unordered_map lookup (C++20 supports view lookup but safe to cast here)
            std::string token_str(token_view);
            
            auto it = vocab_map_.find(token_str);
            if (it != vocab_map_.end()) {
                // Found: Use the index
                // Note: For an Autoencoder/Embedding model, we often pass the ID directly.
                // But since our current 'ParsedLog' expects floats, we normalize the ID 
                // or use One-Hot. For this MVP, let's assume we are populating input features.
                
                // APPROACH A: ID Encoding (if xInfer accepts float inputs acting as IDs)
                out_vector[vector_idx] = static_cast<float>(it->second);
            } else {
                // Not Found: Use [UNK]
                out_vector[vector_idx] = static_cast<float>(unk_token_id_);
            }

            vector_idx++;
        }

        // Fill the rest with Padding [PAD] if necessary
        while (vector_idx < 128) {
            out_vector[vector_idx] = static_cast<float>(pad_token_id_);
            vector_idx++;
        }
    }

} // namespace blackbox::parser