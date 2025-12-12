/**
 * @file block_list_manager.h
 * @brief Dynamic Firewall Controller.
 *
 * Manages the lifecycle of Active Defense blocks.
 * Automatically removes bans after a cooldown period to prevent
 * firewall table exhaustion and permanent false positives.
 */

#ifndef BLACKBOX_ANALYSIS_BLOCK_LIST_MANAGER_H
#define BLACKBOX_ANALYSIS_BLOCK_LIST_MANAGER_H

#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>

namespace blackbox::analysis {

    struct BlockEntry {
        std::string ip;
        std::chrono::steady_clock::time_point start_time;
        int duration_seconds;
    };

    class BlockListManager {
    public:
        // Singleton
        BlockListManager(const BlockListManager&) = delete;
        BlockListManager& operator=(const BlockListManager&) = delete;
        static BlockListManager& instance();

        ~BlockListManager();

        /**
         * @brief Ban an IP address at the OS level.
         *
         * @param ip The IPv4 string
         * @param duration_seconds How long to ban (default 600s = 10 mins)
         */
        void block_ip(const std::string& ip, int duration_seconds = 600);

        /**
         * @brief Manually remove a ban.
         */
        void unblock_ip(const std::string& ip);

        /**
         * @brief Check if an IP is currently blocked by us.
         */
        bool is_blocked(const std::string& ip);

    private:
        BlockListManager();

        /**
         * @brief Background loop that checks for expired bans.
         */
        void expiration_worker();

        /**
         * @brief Executes the actual shell command.
         * @param add True to add rule, False to delete rule.
         */
        void execute_firewall_command(const std::string& ip, bool add);

        // STATE
        std::unordered_map<std::string, BlockEntry> active_blocks_;
        std::mutex mutex_;

        // WORKER
        std::atomic<bool> running_;
        std::thread worker_thread_;
    };

} // namespace blackbox::analysis

#endif // BLACKBOX_ANALYSIS_BLOCK_LIST_MANAGER_H