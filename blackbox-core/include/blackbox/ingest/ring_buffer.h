/**
 * @file ring_buffer.h
 * @brief Definition of the Lock-Free Ring Buffer.
 */

#ifndef BLACKBOX_INGEST_RING_BUFFER_H
#define BLACKBOX_INGEST_RING_BUFFER_H

#include <atomic>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace blackbox::ingest {

    // The data packet moving through the pipeline
    struct LogEvent {
        uint64_t timestamp_ns;
        char raw_data[4096]; // Fixed size 4KB buffer
        size_t length;
    };

    /**
     * @brief Single-Producer Single-Consumer (SPSC) Lock-Free Queue.
     * 
     * @tparam Capacity The size of the buffer (Must be power of 2 for optimization, usually 65536)
     */
    template <size_t Capacity>
    class RingBuffer {
    public:
        RingBuffer();
        ~RingBuffer() = default;

        /**
         * @brief Writer method (Called by UDP Server)
         * @param data Raw bytes
         * @param len Length of bytes
         * @return true if successful, false if full
         */
        bool push(const char* data, size_t len);

        /**
         * @brief Reader method (Called by AI Worker)
         * @param out_event Reference to fill
         * @return true if data exists, false if empty
         */
        bool pop(LogEvent& out_event);

    private:
        // Storage
        std::vector<LogEvent> buffer_;

        // Indices with Cache Padding to prevent False Sharing
        // We align to 64 bytes (common cache line size)
        alignas(64) std::atomic<size_t> head_;
        alignas(64) std::atomic<size_t> tail_;
    };

} // namespace blackbox::ingest

#endif // BLACKBOX_INGEST_RING_BUFFER_H