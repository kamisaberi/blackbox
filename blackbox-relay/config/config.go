package config

import "os"

type Config struct {
	RedisHost       string
	RedisPort       string
	SlackWebhookURL string
	PagerDutyKey    string
}

func Load() *Config {
	return &Config{
		RedisHost:       getEnv("REDIS_HOST", "localhost"),
		RedisPort:       getEnv("REDIS_PORT", "6379"),
		SlackWebhookURL: getEnv("SLACK_WEBHOOK_URL", ""), // https://hooks.slack.com/...
		PagerDutyKey:    getEnv("PAGERDUTY_KEY", ""),
	}
}

func getEnv(key, fallback string) string {
	if val, ok := os.LookupEnv(key); ok {
		return val
	}
	return fallback
}