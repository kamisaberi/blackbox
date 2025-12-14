/**
 * @file main.cpp
 * @brief GoogleTest Entry Point.
 */

#include <gtest/gtest.h>
#include "blackbox/common/logger.h"

int main(int argc, char **argv) {
    // 1. Initialize GoogleTest
    ::testing::InitGoogleTest(&argc, argv);

    // 2. Configure Blackbox Logger for tests
    // We set it to ERROR only so normal INFO logs don't clutter test output
    blackbox::common::Logger::instance().set_level(blackbox::common::LogLevel::ERROR);

    // 3. Run All Tests
    return RUN_ALL_TESTS();
}