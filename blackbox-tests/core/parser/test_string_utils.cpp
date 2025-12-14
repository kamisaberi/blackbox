#include <gtest/gtest.h>
#include "blackbox/common/string_utils.h"

using blackbox::common::StringUtils;

TEST(StringUtilsTest, Trim) {
    EXPECT_EQ(StringUtils::trim("  hello  "), "hello");
    EXPECT_EQ(StringUtils::trim("hello"), "hello");
    EXPECT_EQ(StringUtils::trim("   "), "");
    EXPECT_EQ(StringUtils::trim("\t\nhello\n"), "hello");
}

TEST(StringUtilsTest, Split) {
    auto parts = StringUtils::split("user,pass,ip", ',');
    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "user");
    EXPECT_EQ(parts[1], "pass");
    EXPECT_EQ(parts[2], "ip");

    auto single = StringUtils::split("noseparator", ',');
    ASSERT_EQ(single.size(), 1);
    EXPECT_EQ(single[0], "noseparator");
}

TEST(StringUtilsTest, SQLEscape) {
    std::string input = "O'Reilly";
    std::string expected = "O\\'Reilly";
    EXPECT_EQ(StringUtils::escape_sql(input), expected);

    std::string injection = "admin'; DROP TABLE logs;--";
    std::string safe = StringUtils::escape_sql(injection);
    // Ensure single quote is escaped
    EXPECT_NE(safe.find("\\'"), std::string::npos);
}

TEST(StringUtilsTest, StartsWith) {
    EXPECT_TRUE(StringUtils::starts_with("HELLO AGENT", "HELLO"));
    EXPECT_FALSE(StringUtils::starts_with("HELLO AGENT", "hello")); // Case sensitive
    EXPECT_FALSE(StringUtils::starts_with("HI", "HELLO"));
}