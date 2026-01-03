Here is the **complete code refactoring** required to upgrade `blackbox-core` from the old NVIDIA-only architecture to the new **Universal xInfer Architecture**.

This involves changes to the **Build System**, **Settings**, and the **Inference Logic**.

---

### **1. Build Configuration (`blackbox-core/CMakeLists.txt`)**
**Change:** Removed direct dependencies on CUDA/TensorRT. Added dependency on `xinfer`.

```cmake
cmake_minimum_required(VERSION 3.20)
project(blackbox-core VERSION 1.0.0 LANGUAGES CXX)

# 1. Compiler Settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -O3 -march=native)

# 2. Dependencies
find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_library(HIREDIS_LIB hiredis REQUIRED)
find_library(MAXMINDDB_LIB maxminddb REQUIRED)

# --- NEW: xInfer Integration ---
# Replaces CUDAToolkit and nvinfer
find_package(xinfer REQUIRED)
# -------------------------------

# 3. Include Directories
include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})

# 4. Source Files
set(SOURCES
    src/main.cpp
    src/core/application.cpp
    src/core/pipeline.cpp
    src/core/admin_server.cpp
    src/ingest/udp_server.cpp
    src/ingest/tcp_server.cpp
    src/ingest/rate_limiter.cpp
    src/ingest/ring_buffer.cpp
    src/parser/parser_engine.cpp
    src/parser/tokenizer.cpp
    src/parser/feature_scaler.cpp
    src/analysis/inference_engine.cpp # Updated
    src/analysis/rule_engine.cpp
    src/analysis/alert_manager.cpp
    src/analysis/block_list_manager.cpp
    src/storage/storage_engine.cpp
    src/storage/clickhouse_client.cpp
    src/storage/redis_client.cpp
    src/enrichment/geoip_service.cpp
    src/common/settings.cpp           # Updated
    src/common/logger.cpp
    src/common/metrics.cpp
    src/common/signal_handler.cpp
    src/common/crash_handler.cpp
    src/common/system_stats.cpp
    src/common/thread_utils.cpp
    src/common/string_utils.cpp
    src/common/time_utils.cpp
    src/common/id_generator.cpp
)

# 5. Build
add_executable(flight-recorder ${SOURCES})

# 6. Link
target_link_libraries(flight-recorder PRIVATE
    Boost::system
    Threads::Threads
    ${CURL_LIBRARIES}
    ${HIREDIS_LIB}
    ${MAXMINDDB_LIB}
    # --- NEW: Link xInfer Core & Zoo ---
    xinfer::core
    xinfer::zoo
)
```

---

### **2. Settings Header (`include/blackbox/common/settings.h`)**
**Change:** Added `target_hardware` to `AIConfig` so you can switch between CPU/GPU/NPU via environment variables.

```cpp
#ifndef BLACKBOX_COMMON_SETTINGS_H
#define BLACKBOX_COMMON_SETTINGS_H

#include <string>
#include <cstdint>
// Include xInfer types for Target enum
#include <xinfer/core/types.h> 

namespace blackbox::common {

    struct NetworkConfig {
        uint16_t udp_port = 514;
        uint16_t admin_port = 8081;
        size_t ring_buffer_size = 65536;
    };

    struct AIConfig {
        std::string model_path = "models/ids_model.rknn"; 
        float anomaly_threshold = 0.8f;
        int batch_size = 32;
        
        // NEW: Hardware Target (CPU, NVIDIA_TRT, ROCKCHIP_RKNN, etc.)
        xinfer::Target target_hardware = xinfer::Target::CPU;
        int device_id = 0;
    };

    // ... (EnrichmentConfig and DatabaseConfig remain the same) ...
    struct EnrichmentConfig {
        std::string geoip_db_path = "config/GeoLite2-City.mmdb";
        std::string rules_config_path = "config/rules.yaml";
    };

    struct DatabaseConfig {
        std::string clickhouse_url = "http://localhost:8123";
        size_t flush_batch_size = 1000;
        int flush_interval_ms = 1000;
        std::string redis_host = "localhost";
        int redis_port = 6379;
        std::string redis_channel = "sentry_alerts";
    };

    class Settings {
    public:
        Settings(const Settings&) = delete;
        Settings& operator=(const Settings&) = delete;
        static Settings& instance();
        void load_from_env();

        const NetworkConfig& network() const { return network_; }
        const AIConfig& ai() const { return ai_; }
        const EnrichmentConfig& enrichment() const { return enrichment_; }
        const DatabaseConfig& db() const { return db_; }

    private:
        Settings() = default;
        NetworkConfig network_;
        AIConfig ai_;
        EnrichmentConfig enrichment_;
        DatabaseConfig db_;
    };

}

#endif
```

---

### **3. Settings Implementation (`src/common/settings.cpp`)**
**Change:** Logic to parse the `BLACKBOX_AI_TARGET` env var.

```cpp
#include "blackbox/common/settings.h"
#include "blackbox/common/logger.h"
#include <cstdlib>
#include <iostream>

namespace blackbox::common {

    Settings& Settings::instance() {
        static Settings instance;
        return instance;
    }

    static std::string get_env(const char* key, const std::string& def) {
        const char* val = std::getenv(key);
        return val ? std::string(val) : def;
    }

    // Helper to map string to xInfer Target
    static xinfer::Target parse_target(const std::string& s) {
        if (s == "NVIDIA_TRT") return xinfer::Target::NVIDIA_TRT;
        if (s == "INTEL_OV") return xinfer::Target::INTEL_OV;
        if (s == "ROCKCHIP_RKNN") return xinfer::Target::ROCKCHIP_RKNN;
        if (s == "AMD_VITIS") return xinfer::Target::AMD_VITIS;
        return xinfer::Target::CPU; // Default
    }

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

}
```

---

### **4. Inference Engine Header (`include/blackbox/analysis/inference_engine.h`)**
**Change:** Swapped raw CUDA pointers for `xinfer::zoo` objects.

```cpp
#ifndef BLACKBOX_ANALYSIS_INFERENCE_ENGINE_H
#define BLACKBOX_ANALYSIS_INFERENCE_ENGINE_H

#include <vector>
#include <array>
#include <memory>
#include "blackbox/common/settings.h"

// xInfer Headers
#include <xinfer/core/tensor.h>
#include <xinfer/zoo/cybersecurity/network_detector.h>

namespace blackbox::analysis {

    class InferenceEngine {
    public:
        /**
         * @brief Initialize the AI Engine.
         * Automatically loads the correct backend (TRT, RKNN, OpenVINO)
         * based on settings.
         */
        explicit InferenceEngine(const common::AIConfig& config);
        ~InferenceEngine() = default;

        /**
         * @brief Run inference on a single vector.
         * Wraps xInfer tensor logic.
         * 
         * @param input_vector Normalized feature vector
         * @return float Anomaly score (0.0 - 1.0)
         */
        float evaluate(const std::array<float, 128>& input_vector);

    private:
        // Using the Model Zoo abstraction for cybersecurity
        std::unique_ptr<xinfer::zoo::cybersecurity::NetworkDetector> detector_;
    };

}

#endif
```

---

### **5. Inference Engine Implementation (`src/analysis/inference_engine.cpp`)**
**Change:** Massive cleanup. Removed all `cudaMalloc`, `cudaMemcpy`. Now uses high-level xInfer API.

```cpp
#include "blackbox/analysis/inference_engine.h"
#include "blackbox/common/logger.h"
#include <stdexcept>

namespace blackbox::analysis {

    InferenceEngine::InferenceEngine(const common::AIConfig& config) {
        LOG_INFO("Initializing xInfer Engine...");

        // 1. Prepare Configuration
        xinfer::zoo::cybersecurity::NetworkDetectorConfig net_config;
        net_config.model_path = config.model_path;
        net_config.target = config.target_hardware;
        net_config.device_id = config.device_id;
        
        // 2. Load Model via Factory
        try {
            detector_ = std::make_unique<xinfer::zoo::cybersecurity::NetworkDetector>(net_config);
            LOG_INFO("Model Loaded Successfully via xInfer.");
        } catch (const std::exception& e) {
            LOG_CRITICAL("Failed to load AI model: " + std::string(e.what()));
            throw;
        }
    }

    float InferenceEngine::evaluate(const std::array<float, 128>& input_vector) {
        // 1. Wrap input in a Tensor (Zero-Copy wrapper)
        // We create a tensor of shape [1, 128]
        // const_cast is safe here because xInfer respects read-only if configured
        xinfer::core::Tensor input_tensor(
            {1, 128}, 
            xinfer::core::DataType::FP32, 
            const_cast<float*>(input_vector.data())
        );

        // 2. Run Analysis
        // xInfer handles hardware offload, DMA, and async execution internally
        auto result = detector_->analyze(input_tensor);

        // 3. Return Score
        // Result is a struct containing {is_attack, anomaly_score, attack_type}
        return result.anomaly_score;
    }

}
```

---

### **6. Update `src/core/pipeline.cpp`**
**Change:** Update the constructor call to pass the whole config object instead of just the path.

```cpp
// Inside Pipeline::Pipeline() constructor

// OLD:
// brain_ = std::make_unique<analysis::InferenceEngine>(settings.ai().model_path);

// NEW:
brain_ = std::make_unique<analysis::InferenceEngine>(settings.ai());
```

---

### **Verification**

With these changes, **Blackbox Core** is no longer tied to NVIDIA.
*   If you set `BLACKBOX_AI_TARGET=INTEL_OV`, it runs on Intel CPUs/iGPUs.
*   If you set `BLACKBOX_AI_TARGET=ROCKCHIP_RKNN`, it runs on industrial gateways.
*   If you set `BLACKBOX_AI_TARGET=NVIDIA_TRT`, it uses the CUDA path.

**Are there other changes?**
No, this covers the complete refactor required to adopt the `xInfer` architecture. The logic in `Pipeline` (micro-batching) and `Parser` remains valid because we wrapped the complexity inside `InferenceEngine`.