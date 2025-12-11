/**
 * @file alert_manager.cpp
 * @brief Implementation of Active Defense Logic.
 */

#include "blackbox/analysis/alert_manager.h"
#include "blackbox/common/logger.h" // Our logger
#include "blackbox/common/settings.h" // To check if Active Defense is enabled
#include <cstdlib> // For std::system
#include <iostream>

namespace blackbox::analysis {

    // =========================================================
    // Singleton Instance
    // =========================================================
    AlertManager& AlertManager::instance() {
        static AlertManager instance;
        return instance;
    }

    // =========================================================
    // Helper: Deduplication Logic
    // =========================================================
    bool AlertManager::should_trigger(const std::string& ip) {
        std::lock_guard<std::mutex> lock(map_mutex_);
        
        auto now = std::chrono::steady_clock::now();
        auto it = cooldown_map_.find(ip);

        if (it == cooldown_map_.end()) {
            // New IP, trigger away
            cooldown_map_[ip] = now;
            return true;
        }

        // Check time difference
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - it->second).count();
        
        if (elapsed > COOLDOWN_SECONDS) {
            // Cooldown expired, trigger again
            it->second = now;
            return true;
        }

        // Too soon
        return false;
    }

    // =========================================================
    // Helper: System Blocking (The "Active" part)
    // =========================================================
    void AlertManager::execute_block_action(const std::string& ip) {
        // SAFETY CHECK: Ensure Active Defense is enabled in Settings
        // In a real app, this would be: if (!Settings::instance().is_active_defense_enabled()) return;
        
        // Construct command
        // Example: Add to iptables DROP list
        // WARNING: This requires the container to have CAP_NET_ADMIN capabilities
        std::string cmd = "iptables -A INPUT -s " + ip + " -j DROP";

        LOG_CRITICAL("Executing Active Defense: " + cmd);
        
        // Call system (Blocking call)
        // In production, use fork/exec or a library like 'boost::process' 
        // to avoid stalling the thread.
        int ret = std::system(cmd.c_str());
        
        if (ret != 0) {
            LOG_ERROR("Failed to execute block command for IP: " + ip);
        } else {
            LOG_INFO("Successfully blocked IP: " + ip);
        }
    }

    // =========================================================
    // Trigger Alert (The Hot Path)
    // =========================================================
    void AlertManager::trigger_alert(std::string_view source_ip, float score, std::string_view message) {
        
        // 1. Check Threshold
        // Note: The caller usually checks this, but we double-check for safety
        if (score < 0.8f) return;

        std::string ip_str(source_ip);

        // 2. Deduplicate
        if (!should_trigger(ip_str)) {
            // We know about this threat, silently ignore to save resources
            return;
        }

        // 3. Log to Console (Red)
        std::string msg = "THREAT DETECTED [Score: " + std::to_string(score) + "] IP: " + ip_str;
        LOG_CRITICAL(msg);

        // 4. Active Defense (If Critical)
        if (score > CRITICAL_THRESHOLD) {
            // Only block if it's practically 100% certain (0.95+)
            // We don't want to block users on false positives.
            execute_block_action(ip_str);
        }
    }

} // namespace blackbox::analysis