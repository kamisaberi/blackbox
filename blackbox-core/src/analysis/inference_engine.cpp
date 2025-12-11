/**
 * @file inference_engine.cpp
 * @brief Implementation of the GPU Bridge.
 */

#include "blackbox/analysis/inference_engine.h"
#include <iostream>
#include <stdexcept>
#include <cstring> // memcpy

// Assuming your proprietary library headers look like this:
#include "xinfer/engine.h"
#include "xinfer/context.h"
#include <cuda_runtime.h> // Standard CUDA API

namespace blackbox::analysis {

    // =========================================================
    // Constructor
    // =========================================================
    InferenceEngine::InferenceEngine(const std::string& model_path) 
        : d_input_(nullptr), d_output_(nullptr) 
    {
        std::cout << "[CORE] Loading AI Model from: " << model_path << "..." << std::endl;

        // 1. Load the xInfer Engine (Wrapper around nvinfer1::ICudaEngine)
        try {
            engine_ = xInfer::load_engine(model_path);
        } catch (const std::exception& e) {
            std::cerr << "[ERR] Failed to load model: " << e.what() << std::endl;
            throw;
        }

        // 2. Create Execution Context
        context_ = engine_->create_execution_context();

        // 3. Allocate GPU Memory
        // Assuming 128 floats input, 1 float output
        input_size_bytes_ = 128 * sizeof(float);
        output_size_bytes_ = 1 * sizeof(float);

        cudaError_t err;
        
        err = cudaMalloc(&d_input_, input_size_bytes_);
        if (err != cudaSuccess) throw std::runtime_error("CUDA Malloc Input Failed");

        err = cudaMalloc(&d_output_, output_size_bytes_);
        if (err != cudaSuccess) throw std::runtime_error("CUDA Malloc Output Failed");

        std::cout << "[CORE] Model Loaded Successfully. GPU Ready." << std::endl;
    }

    // =========================================================
    // Destructor
    // =========================================================
    InferenceEngine::~InferenceEngine() {
        // Free GPU memory
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
    }

    // =========================================================
    // Evaluate (The Hot Path)
    // =========================================================
    float InferenceEngine::evaluate(const std::array<float, 128>& input_vector) {
        
        // 1. Host -> Device Copy (CPU to GPU)
        // In a batched system, we would copy 32 vectors at once here.
        cudaMemcpy(d_input_, input_vector.data(), input_size_bytes_, cudaMemcpyHostToDevice);

        // 2. Run Inference
        // Use your xInfer library's execute wrapper
        // Usually takes an array of bindings [input_ptr, output_ptr]
        std::vector<void*> bindings = { d_input_, d_output_ };
        
        bool success = context_->execute(bindings);
        if (!success) {
            std::cerr << "[ERR] xInfer execution failed!" << std::endl;
            return 0.0f; // Fail safe (assume benign to prevent blocking traffic on error)
        }

        // 3. Device -> Host Copy (GPU to CPU)
        float anomaly_score = 0.0f;
        cudaMemcpy(&anomaly_score, d_output_, output_size_bytes_, cudaMemcpyDeviceToHost);

        // 4. Return Score
        return anomaly_score;
    }

} // namespace blackbox::analysis