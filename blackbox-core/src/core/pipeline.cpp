/**
 * @file pipeline.cpp
 * @brief Implementation of the Orchestrator.
 */

#include "blackbox/core/pipeline.h"
#include "blackbox/common/settings.h"
#include "blackbox/common/logger.h"
#include "blackbox/common/metrics.h"
#include "blackbox/analysis/alert_manager.h"
#include <iostream>
#include <chrono>

namespace blackbox::core {

    // =========================================================
    // Constructor
    // =========================================================
    Pipeline::Pipeline() {
        LOG_INFO("Initializing Blackbox Pipeline...");

        // Load Settings
        const auto& settings = common::Settings::instance();

        // 1. Setup Network Context
        io_context_ = std::make_shared<boost::asio::io_context>();
        
        // 2. Setup UDP Server
        udp_server_ = std::make_unique<ingest::UdpServer>(
            *io_context_, 
            settings.network().udp_port, 
            ring_buffer_
        );

        // 3. Setup AI
        try {
            brain_ = std::make_unique<analysis::InferenceEngine>(settings.ai().model_path);
        } catch (const std::exception& e) {
            LOG_CRITICAL("Failed to load AI Brain. System is blind.");
            throw; // Fatal error
        }
    }

    Pipeline::~Pipeline() {
        stop();
    }

    // =========================================================
    // Start / Stop
    // =========================================================
    void Pipeline::start() {
        if (running_) return;
        running_ = true;

        LOG_INFO("Spawning Worker Threads...");

        // Start Network Thread
        ingest_thread_ = std::thread(&Pipeline::ingest_worker, this);

        // Start CPU/GPU Worker
        processing_thread_ = std::thread(&Pipeline::processing_worker, this);

        LOG_INFO("Pipeline Active. Kinetic Defense Online.");
    }

    void Pipeline::stop() {
        if (!running_) return;
        LOG_WARN("Stopping Pipeline...");
        
        running_ = false;

        // Stop Network IO
        if (io_context_) io_context_->stop();

        // Join Threads
        if (ingest_thread_.joinable()) ingest_thread_.join();
        if (processing_thread_.joinable()) processing_thread_.join();

        LOG_INFO("Pipeline Stopped.");
    }

    bool Pipeline::is_healthy() const {
        return running_;
    }

    // =========================================================
    // Worker 1: Ingestion (Network IO)
    // =========================================================
    void Pipeline::ingest_worker() {
        // High-priority thread setting could go here
        try {
            io_context_->run(); // Infinite loop until stop() is called
        } catch (const std::exception& e) {
            LOG_CRITICAL("Ingestion Thread Crashed: " + std::string(e.what()));
        }
    }

    // =========================================================
    // Worker 2: Processing (The Hot Path)
    // =========================================================
    void Pipeline::processing_worker() {
        const int BATCH_SIZE = common::Settings::instance().ai().batch_size;
        
        // Local buffers to avoid allocation in the loop
        std::vector<parser::ParsedLog> batch_logs;
        batch_logs.reserve(BATCH_SIZE);
        
        // Vector for AI Input (Flattened: BATCH_SIZE * 128)
        // In a real scenario, InferenceEngine would accept this flat vector
        // For this MVP code, we simulate looping the single-item inference.

        ingest::LogEvent raw_event;

        while (running_) {
            // ==========================================
            // 1. MICRO-BATCHING LOGIC
            // ==========================================
            // Try to fill the batch from the RingBuffer
            int collected = 0;
            while (collected < BATCH_SIZE && ring_buffer_.pop(raw_event)) {
                // Parse immediately (Zero Copy)
                batch_logs.push_back(parser_.process(raw_event));
                collected++;
            }

            // If buffer is empty and we have no data, sleep briefly to save CPU
            if (collected == 0) {
                std::this_thread::yield(); // or sleep_for(1us)
                continue;
            }

            // ==========================================
            // 2. INFERENCE
            // ==========================================
            
            // NOTE: In production, pass the whole 'batch_logs' to brain_->evaluate_batch()
            // Here we loop for the MVP interface
            for (const auto& log : batch_logs) {
                float score = brain_->evaluate(log.embedding_vector);

                // Update Metrics
                common::Metrics::instance().inc_inferences_run(1);

                // ==========================================
                // 3. DECISION & ROUTING
                // ==========================================
                
                // A. Active Defense
                if (score > common::Settings::instance().ai().anomaly_threshold) {
                    common::Metrics::instance().inc_threats_detected(1);
                    
                    // Convert string_view to string for safety in AlertManager
                    analysis::AlertManager::instance().trigger_alert(
                        log.host, score, log.message
                    );
                }

                // B. Persistence
                storage_.enqueue(log, score);
            }

            // ==========================================
            // 4. CLEANUP
            // ==========================================
            // Clear vector but keep memory reserved (capacity stays)
            batch_logs.clear(); 
        }
    }

} // namespace blackbox::core