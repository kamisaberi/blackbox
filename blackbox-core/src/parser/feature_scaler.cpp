/**
 * @file feature_scaler.cpp
 * @brief Implementation of MinMax Scaling.
 */

#include "blackbox/parser/feature_scaler.h"
#include "blackbox/common/logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace blackbox::parser {

    // =========================================================
    // Constructor
    // =========================================================
    FeatureScaler::FeatureScaler() {
        // Reserve standard size to avoid realloc
        min_vals_.reserve(128);
        scale_factors_.reserve(128);
    }

    // =========================================================
    // Load Parameters
    // =========================================================
    bool FeatureScaler::load_parameters(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open scaler parameters: " + path);
            return false;
        }

        min_vals_.clear();
        scale_factors_.clear();

        std::string line;
        int line_num = 0;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string segment;
            std::vector<float> vals;

            // Parse CSV line "min,max"
            while (std::getline(ss, segment, ',')) {
                try {
                    vals.push_back(std::stof(segment));
                } catch (...) {
                    LOG_WARN("Invalid scaler param at line " + std::to_string(line_num));
                }
            }

            if (vals.size() >= 2) {
                float min_v = vals[0];
                float max_v = vals[1];
                
                // Store Min
                min_vals_.push_back(min_v);

                // Pre-calculate Scale Factor (Optimization)
                // Division is expensive in CPU cycles. Multiplication is cheap.
                // We compute 1 / (max - min) now, so we can multiply later.
                float range = max_v - min_v;
                if (std::abs(range) < 1e-6) {
                    scale_factors_.push_back(0.0f); // Avoid divide by zero
                } else {
                    scale_factors_.push_back(1.0f / range);
                }
            }
            line_num++;
        }

        if (min_vals_.size() != 128) {
            LOG_WARN("Scaler loaded " + std::to_string(min_vals_.size()) + " params, expected 128.");
        } else {
            LOG_INFO("Feature Scaler ready.");
        }
        
        ready_ = true;
        return true;
    }

    // =========================================================
    // Transform (The Hot Path)
    // =========================================================
    void FeatureScaler::transform(std::array<float, 128>& vector) const {
        if (!ready_) return;

        // Ensure we don't go out of bounds if params mismatch vector size
        size_t limit = std::min(vector.size(), min_vals_.size());

        // This loop is a prime candidate for auto-vectorization (AVX2/AVX512)
        // by the C++ compiler (O3 flag).
        for (size_t i = 0; i < limit; ++i) {
            float val = vector[i];
            
            // Formula: (val - min) / (max - min)
            // Optimization: (val - min) * factor
            val = (val - min_vals_[i]) * scale_factors_[i];

            // Clamp between 0.0 and 1.0 (Safety net for outliers)
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;

            vector[i] = val;
        }
    }

} // namespace blackbox::parser