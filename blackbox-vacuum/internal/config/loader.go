package config

import (
	"os"
	"strconv"
)

type Config struct {
	// Core Connection
	CoreHost string
	CorePort int

	// Webhook Server
	WebhookPort string

	// Service Credentials (loaded from ENV)
	AWSRegion     string
	MSSQLConn     string
	SlackToken    string
	OktaDomain    string
	OktaAPIToken  string
}

func Load() *Config {
	return &Config{
		CoreHost:    getEnv("VACUUM_CORE_HOST", "blackbox-core"),
		CorePort:    getEnvInt("VACUUM_CORE_PORT", 601),
		WebhookPort: getEnv("VACUUM_WEBHOOK_PORT", "9090"),

		// Service Specifics
		AWSRegion:    getEnv("AWS_REGION", "us-east-1"),
		MSSQLConn:    getEnv("MSSQL_CONN_STRING", ""), // e.g., sqlserver://sa:pass@host:1433
		SlackToken:   getEnv("SLACK_API_TOKEN", ""),
		OktaDomain:   getEnv("OKTA_DOMAIN", ""),
		OktaAPIToken: getEnv("OKTA_API_TOKEN", ""),
	}
}

// Helper: Get string env
func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

// Helper: Get int env
func getEnvInt(key string, fallback int) int {
	strValue := getEnv(key, "")
	if strValue == "" {
		return fallback
	}
	val, err := strconv.Atoi(strValue)
	if err != nil {
		return fallback
	}
	return val
}