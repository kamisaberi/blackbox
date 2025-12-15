#include <gtest/gtest.h>
#include "blackbox/ingest/ring_buffer.h"
#include <string>

// Test Fixture
class RingBufferTest : public ::testing::Test {
protected:
    blackbox::ingest::RingBuffer<16> buffer; // Small buffer for testing
    blackbox::ingest::LogEvent event;
};

TEST_F(RingBufferTest, ShouldStartEmpty) {
    EXPECT_FALSE(buffer.pop(event));
}

TEST_F(RingBufferTest, ShouldPushAndPop) {
    std::string msg = "TestLog";
    EXPECT_TRUE(buffer.push(msg.c_str(), msg.length()));

    EXPECT_TRUE(buffer.pop(event));
    std::string result(event.raw_data, event.length);
    EXPECT_EQ(result, "TestLog");
}

TEST_F(RingBufferTest, ShouldDropWhenFull) {
    std::string msg = "A";

    // Fill buffer (Capacity - 1 is typically usable in lock-free ring buffers)
    for(int i=0; i<16; i++) {
        buffer.push(msg.c_str(), 1);
    }

    // This push should fail
    EXPECT_FALSE(buffer.push(msg.c_str(), 1));
}