package main_test

import (
	"os"
	"testing"

	"blackbox-tower/internal/config"
)

func TestConfigDefaults(t *testing.T) {
	// 1. Unset env vars to test defaults
	os.Unsetenv("TOWER_PORT")
	os.Unsetenv("BLACKBOX_CLICKHOUSE_URL")

	cfg := config.LoadConfig()

	if cfg.ServerPort != "8080" {
		t.Errorf("Expected default port 8080, got %s", cfg.ServerPort)
	}
	if cfg.ClickHouseURL != "tcp://localhost:9000" {
		t.Errorf("Expected default CH URL, got %s", cfg.ClickHouseURL)
	}
}

func TestConfigOverrides(t *testing.T) {
	// 2. Set env vars
	os.Setenv("TOWER_PORT", "9999")
	os.Setenv("JWT_SECRET", "supersecret")

	cfg := config.LoadConfig()

	if cfg.ServerPort != "9999" {
		t.Errorf("Expected port 9999, got %s", cfg.ServerPort)
	}
	if cfg.JWTSecret != "supersecret" {
		t.Errorf("Expected overridden secret, got %s", cfg.JWTSecret)
	}
}