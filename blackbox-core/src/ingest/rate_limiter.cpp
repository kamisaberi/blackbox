/**
 * @file rate_limiter.cpp
 * @brief Implementation of Per-IP Rate Limiting.
 */

#include "blackbox/ingest/rate_limiter.h"
#include "blackbox/common/logger.h"
#include <vector>

namespace blackbox::ingest {

    // =========================================================
    // Singleton
    // =========================================================
    RateLimiter& RateLimiter::instance() {
        static RateLimiter instance;
        return instance;
    }

    RateLimiter::RateLimiter() {
        // Reserve space to minimize re-hashing during startup
        buckets_.reserve(1000);
    }

    // =========================================================
    // Check Permission (Token Bucket Logic)
    // =========================================================
    bool RateLimiter::should_allow(std::string_view ip_address) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::string ip(ip_address);
        auto now = std::chrono::steady_clock::now();

        // 1. Get or Create Bucket
        auto it = buckets_.find(ip);
        if (it == buckets_.end()) {
            TokenBucket new_bucket;
            new_bucket.tokens = DEFAULT_BURST; // Start full
            new_bucket.max_burst = DEFAULT_BURST;
            new_bucket.refill_rate = DEFAULT_RATE;
            new_bucket.last_refill = now;
            
            buckets_[ip] = new_bucket;
            it = buckets_.find(ip);
        }

        TokenBucket& bucket = it->second;

        // 2. Refill Tokens based on time elapsed
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - bucket.last_refill).count();
        double seconds = duration / 1000000.0;
        
        // Add tokens: (Time * Rate)
        double new_tokens = seconds * bucket.refill_rate;
        
        if (new_tokens > 0) {
            bucket.tokens = std::min(bucket.max_burst, bucket.tokens + new_tokens);
            bucket.last_refill = now;
        }

        // 3. Consume Token
        if (bucket.tokens >= 1.0) {
            bucket.tokens -= 1.0;
            return true; // Allowed
        } else {
            return false; // Rate Limited
        }
    }

    // =========================================================
    // Cleanup (Garbage Collection)
    // =========================================================
    void RateLimiter::cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        std::vector<std::string> to_remove;

        for (const auto& [ip, bucket] : buckets_) {
            // If IP hasn't been seen in 60 seconds, remove it
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - bucket.last_refill).count();
            if (elapsed > 60) {
                to_remove.push_back(ip);
            }
        }

        for (const auto& ip : to_remove) {
            buckets_.erase(ip);
        }
        
        if (!to_remove.empty()) {
            LOG_DEBUG("RateLimiter cleaned up " + std::to_string(to_remove.size()) + " inactive IPs.");
        }
    }

} // namespace blackbox::ingest