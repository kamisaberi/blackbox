/**
 * @file model_loader.cpp
 * @brief Implementation of Binary Loader.
 */

#include "blackbox/analysis/model_loader.h"
#include "blackbox/common/logger.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace blackbox::analysis {

    // =========================================================
    // Check Existence
    // =========================================================
    bool ModelLoader::exists(const std::string& path) {
        return std::filesystem::exists(path);
    }

    // =========================================================
    // Get Size
    // =========================================================
    size_t ModelLoader::get_size(const std::string& path) {
        try {
            return std::filesystem::file_size(path);
        } catch (const std::filesystem::filesystem_error& e) {
            LOG_ERROR("Failed to get size of model: " + path + " | " + e.what());
            return 0;
        }
    }

    // =========================================================
    // Load Binary (The Heavy Lift)
    // =========================================================
    std::vector<char> ModelLoader::load_binary(const std::string& path) {
        // 1. Validation
        if (!exists(path)) {
            std::string err = "Model file not found: " + path;
            LOG_CRITICAL(err);
            throw std::runtime_error(err);
        }

        // 2. Open Stream
        // std::ios::binary is CRITICAL. Without it, Windows/Linux newline 
        // conversions will corrupt the neural network weights.
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        
        if (!file.good()) {
            std::string err = "Failed to open model stream: " + path;
            LOG_ERROR(err);
            throw std::runtime_error(err);
        }

        // 3. Determine Size
        // We opened with 'ate' (At The End), so tellg() gives us size
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg); // Go back to start

        if (size <= 0) {
            std::string err = "Model file is empty: " + path;
            LOG_ERROR(err);
            throw std::runtime_error(err);
        }

        LOG_INFO("Loading Model into RAM (" + std::to_string(size) + " bytes): " + path);

        // 4. Allocate Buffer
        std::vector<char> buffer(size);

        // 5. Read
        if (!file.read(buffer.data(), size)) {
            std::string err = "Error reading model data: " + path;
            LOG_ERROR(err);
            throw std::runtime_error(err);
        }

        LOG_INFO("Model loaded successfully.");
        return buffer;
    }

} // namespace blackbox::analysis