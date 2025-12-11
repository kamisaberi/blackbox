/**
 * @file rule_engine.cpp
 * @brief Implementation of Static Signature Matching.
 */

#include "blackbox/analysis/rule_engine.h"
#include "blackbox/common/logger.h"
#include <algorithm>

namespace blackbox::analysis {

    // =========================================================
    // Constructor
    // =========================================================
    RuleEngine::RuleEngine() {
        // In a real app, call load_rules() here.
        // For MVP, we hardcode some sanity rules.
        
        Rule ssh_brute;
        ssh_brute.name = "SSH_FAIL_ROOT";
        ssh_brute.description = "Root login failure";
        ssh_brute.action = RuleAction::ALERT;
        ssh_brute.field_target = "message";
        ssh_brute.pattern = "Failed password for root";
        ssh_brute.is_regex = false;
        
        rules_.push_back(ssh_brute);

        Rule firewall_drop;
        firewall_drop.name = "FW_DROP_TRAFFIC";
        firewall_drop.description = "Firewall blocked packet";
        firewall_drop.action = RuleAction::TAG;
        firewall_drop.field_target = "message";
        firewall_drop.pattern = "BLOCK";
        firewall_drop.is_regex = false;

        rules_.push_back(firewall_drop);

        LOG_INFO("Rule Engine initialized with " + std::to_string(rules_.size()) + " hardcoded rules.");
    }

    // =========================================================
    // Load Rules (Stub)
    // =========================================================
    void RuleEngine::load_rules(const std::string& config_path) {
        // TODO: Implement YAML parser logic here using yaml-cpp
        LOG_INFO("Loading rules from: " + config_path);
    }

    // =========================================================
    // Match Helper
    // =========================================================
    bool RuleEngine::match_condition(std::string_view value, const Rule& rule) {
        if (rule.is_regex) {
            // regex matching (skip for MVP performance reasons, string find is 100x faster)
            return false;
        }
        
        // Substring search (Fast)
        // std::string_view::find is efficient
        return value.find(rule.pattern) != std::string_view::npos;
    }

    // =========================================================
    // Evaluate (The Hot Path)
    // =========================================================
    std::optional<std::string> RuleEngine::evaluate(const parser::ParsedLog& log) {
        
        for (const auto& rule : rules_) {
            bool matched = false;

            // 1. Select Field
            if (rule.field_target == "service") {
                matched = match_condition(log.service, rule);
            } 
            else if (rule.field_target == "host") {
                matched = match_condition(log.host, rule);
            }
            else if (rule.field_target == "message") {
                matched = match_condition(log.message, rule);
            }

            // 2. Return Result
            if (matched) {
                // Return the first match (Short-circuit)
                // In advanced SIEMs, you might want to match ALL rules
                return rule.name;
            }
        }

        return std::nullopt;
    }

} // namespace blackbox::analysis