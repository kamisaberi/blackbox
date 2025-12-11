/**
 * @file ring_buffer.h
 * @brief High-performance, Lock-Free Single-Producer Single-Consumer (SPSC) Queue.
 * 
 * This is the central nervous system of Blackbox. 
 * The UDP Server (Producer) pushes raw logs here.
 * The AI Engine (Consumer) pops logs from here.
 * 
 * OPTIMIZATIONS:
 * 1. Lock-Free: Uses std::atomic for head/tail indices.
 * 2. Cache-Aligned: Prevents "False Sharing" by padding indices to 64 bytes.
 * 3. Pre-Allocated: Zero mallocs during runtime.
 */

#ifndef BLACKBOX_INGEST_RING_BUFFER_H
#define BLACKBOX_INGEST_RING_BUFFER_H

#include <atomic>
#include <vector>
#include <optional>
#include <iostream>
#include <new> // for std::hardware_destructive_interference_size

namespace blackbox::ingest {

    // 1. Define the Data Packet
    // We stick to a fixed size char array to avoid heap allocation (std::string)
    struct LogEvent {
        uint64_t timestamp_ns;
        char raw_data[4096]; // Max log size 4KB
        size_t length;
    };

    template <size_t Capacity>
    class RingBuffer {
    public:
        RingBuffer() : head_(0), tail_(0) {
            // Pre-allocate vector storage to avoid runtime resize
            buffer_.resize(Capacity);
        }

        // ==========================================
        // PRODUCER API (Called by UDP Server Thread)
        // ==========================================
        
        /**
         * @brief Attempts to write a log into the buffer.
         * @return true if successful, false if buffer is full (drop packet).
         */
        bool push(const char* data, size_t len) {
            const size_t current_head = head_.load(std::memory_order_relaxed);
            const size_t next_head = (current_head + 1) % Capacity;

            // Check if full: strictly compare against cached tail
            // memory_order_acquire ensures we see the latest tail update from Consumer
            if (next_head == tail_.load(std::memory_order_acquire)) {
                return false; // Buffer Full - Packet Dropped (Strategy: Drop vs Block)
            }

            // Zero-Copy Optimization:
            // Instead of creating a LogEvent and copying it, we write directly 
            // into the pre-allocated slot memory.
            LogEvent& slot = buffer_[current_head];
            slot.timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            slot.length = (len > 4096) ? 4096 : len;
            std::memcpy(slot.raw_data, data, slot.length);

            // Publish the change
            // memory_order_release ensures the data write completes BEFORE we update the index
            head_.store(next_head, std::memory_order_release);
            return true;
        }

        // ==========================================
        // CONSUMER API (Called by AI/Parser Thread)
        // ==========================================

        /**
         * @brief Checks if there is data to process.
         * @param out_event Reference to fill with data if available.
         * @return true if data was popped, false if empty.
         */
        bool pop(LogEvent& out_event) {
            const size_t current_tail = tail_.load(std::memory_order_relaxed);

            // Check if empty
            // memory_order_acquire ensures we see the latest head update from Producer
            if (current_tail == head_.load(std::memory_order_acquire)) {
                return false; // Buffer Empty
            }

            // Read the data
            // Copy from slot to output (Cheap copy for structs)
            out_event = buffer_[current_tail];

            // Advance tail
            const size_t next_tail = (current_tail + 1) % Capacity;
            tail_.store(next_tail, std::memory_order_release);
            return true;
        }

    private:
        // STORAGE
        std::vector<LogEvent> buffer_;

        // INDICES (Atomic & Cache Aligned)
        
        // alignas(64) ensures this variable sits on its own Cache Line (L1/L2/L3).
        // Without this, if 'head' and 'tail' share a cache line, the CPU cores 
        // fight over ownership (Cache Thrashing), slowing down throughput by 50x.
        
        alignas(64) std::atomic<size_t> head_; // Modified by Producer
        alignas(64) std::atomic<size_t> tail_; // Modified by Consumer
    };

} // namespace blackbox::ingest

#endif // BLACKBOX_INGEST_RING_BUFFER_H