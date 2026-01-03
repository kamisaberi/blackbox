/**
 * @file settings.cpp
 * @brief Implementation of Configuration Loading.
 */

#include "blackbox/common/settings.h"
#include <cstdlib> // for std::getenv
#include <iostream>
#include <string>

namespace blackbox::common {

    // =========================================================
    // Singleton Instance
    // =========================================================
    Settings& Settings::instance() {
        static Settings instance;
        return instance;
    }

    // =========================================================
    // Helper: Parse Env Var safely
    // =========================================================
    static std::string get_env_string(const char* key, const std::string& default_val) {
        const char* val = std::getenv(key);
        return val ? std::string(val) : default_val;
    }

    static int get_env_int(const char* key, int default_val) {
        const char* val = std::getenv(key);
        if (val) {
            try {
                return std::stoi(val);
            } catch (...) {
                std::cerr << "[WARN] Invalid integer for ENV " << key << ", using default." << std::endl;
            }
        }
        return default_val;
    }

    static float get_env_float(const char* key, float default_val) {
        const char* val = std::getenv(key);
        if (val) {
            try {
                return std::stof(val);
            } catch (...) {
                std::cerr << "[WARN] Invalid float for ENV " << key << ", using default." << std::endl;
            }
        }
        return default_val;
    }


    // Helper to map string to xInfer Target
    static xinfer::Target parse_target(const std::string& s) {
        if (s == "NVIDIA_TRT") return xinfer::Target::NVIDIA_TRT;
        if (s == "INTEL_OV") return xinfer::Target::INTEL_OV;
        if (s == "ROCKCHIP_RKNN") return xinfer::Target::ROCKCHIP_RKNN;
        if (s == "AMD_VITIS") return xinfer::Target::AMD_VITIS;
        return xinfer::Target::CPU; // Default
    }


    // =========================================================
    // Load Logic
    // =========================================================
    void Settings::load_from_env() {
        LOG_INFO("Loading Configuration...");

        // Network
        network_.udp_port = std::stoi(get_env("BLACKBOX_UDP_PORT", "514"));
        network_.admin_port = std::stoi(get_env("BLACKBOX_ADMIN_PORT", "8081"));

        // AI Configuration (Updated)
        ai_.model_path = get_env("BLACKBOX_MODEL_PATH", "models/autoencoder.onnx");
        ai_.anomaly_threshold = std::stof(get_env("BLACKBOX_ANOMALY_THRESHOLD", "0.8"));
        ai_.batch_size = std::stoi(get_env("BLACKBOX_AI_BATCH_SIZE", "32"));
        ai_.target_hardware = parse_target(get_env("BLACKBOX_AI_TARGET", "CPU"));
        ai_.device_id = std::stoi(get_env("BLACKBOX_AI_DEVICE_ID", "0"));

        // DB & Enrichment (Standard)
        db_.clickhouse_url = get_env("BLACKBOX_CLICKHOUSE_URL", "http://localhost:8123");
        db_.redis_host = get_env("BLACKBOX_REDIS_HOST", "localhost");
        enrichment_.geoip_db_path = get_env("BLACKBOX_GEOIP_PATH", "config/GeoLite2-City.mmdb");
        enrichment_.rules_config_path = get_env("BLACKBOX_RULES_PATH", "config/rules.yaml");

        LOG_INFO("AI Target: " + xinfer::to_string(ai_.target_hardware));
    }

} // namespace blackbox::common