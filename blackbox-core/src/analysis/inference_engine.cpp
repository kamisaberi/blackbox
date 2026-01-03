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