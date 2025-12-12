/**
 * @file id_generator.h
 * @brief High-performance UUID Generator.
 *
 * Generates version 4 UUIDs for event correlation.
 * Uses thread-local random engines to avoid mutex contention
 * in high-throughput pipelines.
 */

#ifndef BLACKBOX_COMMON_ID_GENERATOR_H
#define BLACKBOX_COMMON_ID_GENERATOR_H

#include <string>

namespace blackbox::common {

    class IdGenerator {
    public:
        /**
         * @brief Generates a random UUID v4 string.
         *
         * Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
         * where x is any hex digit and y is one of 8, 9, A, or B.
         *
         * @return 36-character string
         */
        static std::string generate_uuid_v4();
    };

} // namespace blackbox::common

#endif // BLACKBOX_COMMON_ID_GENERATOR_H