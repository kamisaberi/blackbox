/**
 * @file pipeline.h
 * @brief The Main Data Processing Engine.
 */

#ifndef BLACKBOX_CORE_PIPELINE_H
#define BLACKBOX_CORE_PIPELINE_H

#include <vector>
#include <thread>
#include <atomic>
#include <memory>

// Ingestion
#include "blackbox/ingest/ring_buffer.h"
#include "blackbox/ingest/udp_server.h"
#include "blackbox/ingest/tcp_server.h"

// Logic & Analysis
#include "blackbox/parser/parser_engine.h"
#include "blackbox/analysis/inference_engine.h"
#include "blackbox/analysis/rule_engine.h"
#include "blackbox/enrichment/geoip_service.h"

// Storage & Output
#include "blackbox/storage/storage_engine.h"
#include "blackbox/storage/redis_client.h"

// Ops
#include "blackbox/core/admin_server.h"

#include "blackbox/ingest/netflow_server.h"

namespace blackbox::core {

    class Pipeline {
    public:
        Pipeline();
        ~Pipeline();

        void start();
        void stop();
        bool is_healthy() const;

    private:
        // Thread Functions
        void ingest_worker();
        void processing_worker();

        // State
        std::atomic<bool> running_{false};
        std::thread ingest_thread_;
        std::thread processing_thread_;


        std::unique_ptr<ingest::NetflowServer> netflow_server_;



        // --- COMPONENTS ---

        // 1. Shared Memory
        ingest::RingBuffer<65536> ring_buffer_;

        // 2. Network Inputs
        std::shared_ptr<boost::asio::io_context> io_context_;
        std::unique_ptr<ingest::UdpServer> udp_server_;
        std::unique_ptr<ingest::TcpServer> tcp_server_;

        std::unique_ptr<AdminServer> admin_server_;

        // 3. The Brains
        parser::ParserEngine parser_;
        std::unique_ptr<analysis::InferenceEngine> brain_;
        std::unique_ptr<analysis::RuleEngine> rule_engine_;
        std::unique_ptr<enrichment::GeoIPService> geoip_;

        // 4. Persistence & Notifications
        storage::StorageEngine storage_;
        std::unique_ptr<storage::RedisClient> redis_;
    };

} // namespace blackbox::core

#endif // BLACKBOX_CORE_PIPELINE_H