#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include "blackbox/parser/parser_engine.h"
#include "blackbox/common/settings.h"

class ParserEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 1. Create dummy config directories
        std::filesystem::create_directories("config");

        // 2. Create dummy vocab.txt
        std::ofstream vocab("config/vocab.txt");
        vocab << "failed\npassword\nroot\n"; // 0, 1, 2
        vocab.close();

        // 3. Create dummy scaler_params.txt (128 lines)
        std::ofstream scaler("config/scaler_params.txt");
        for(int i=0; i<128; i++) scaler << "0.0,1.0\n";
        scaler.close();

        // 4. Update Settings singleton to point to these files
        // (Assuming we can modify settings via env vars or direct access,
        //  here we rely on defaults matching 'config/')
    }

    void TearDown() override {
        std::filesystem::remove_all("config");
    }
};

TEST_F(ParserEngineTest, ParseSyslog) {
    blackbox::parser::ParserEngine parser;
    blackbox::ingest::LogEvent event;

    // Mock Syslog Message: <PRI>VER TS HOST APP PID MSGID SD MSG
    std::string raw_log = "<34>1 2023-10-10T00:00:00Z my-laptop sshd - - - Failed password for root";

    // Copy to event buffer
    std::memcpy(event.raw_data, raw_log.c_str(), raw_log.size());
    event.length = raw_log.size();
    event.timestamp_ns = 1000;

    auto result = parser.process(event);

    EXPECT_EQ(result.host, "my-laptop");
    EXPECT_EQ(result.service, "sshd");
    EXPECT_EQ(result.message, "Failed password for root");
    EXPECT_FALSE(result.id.empty()); // UUID generated
}

TEST_F(ParserEngineTest, ParseGarbage) {
    blackbox::parser::ParserEngine parser;
    blackbox::ingest::LogEvent event;

    std::string raw_log = "Just random garbage text";
    std::memcpy(event.raw_data, raw_log.c_str(), raw_log.size());
    event.length = raw_log.size();

    auto result = parser.process(event);

    // Should fallback
    EXPECT_EQ(result.service, "raw");
    EXPECT_EQ(result.message, "Just random garbage text");
}