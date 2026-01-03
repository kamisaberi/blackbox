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