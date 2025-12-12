package storage

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"

	"blackbox-tower/pkg/models"

	// Import ClickHouse driver
	_ "github.com/ClickHouse/clickhouse-go/v2"
)

type Repository struct {
	db *sql.DB
}

// NewRepository creates a connection pool
func NewRepository(dsn string) (*Repository, error) {
	db, err := sql.Open("clickhouse", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open clickhouse connection: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping clickhouse: %w", err)
	}

	log.Println("Connected to ClickHouse DB")
	return &Repository{db: db}, nil
}

// Close closes the connection
func (r *Repository) Close() {
	if r.db != nil {
		r.db.Close()
	}
}

// GetRecentLogs fetches the latest logs for the Live View
func (r *Repository) GetRecentLogs(limit int) ([]models.LogEntry, error) {
	query := `
		SELECT id, timestamp, host, country, service, message, anomaly_score, is_threat
		FROM sentry.logs
		ORDER BY timestamp DESC
		LIMIT ?
	`

	rows, err := r.db.Query(query, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var logs []models.LogEntry
	for rows.Next() {
		var l models.LogEntry
		// ClickHouse driver handles mapping to Go types
		if err := rows.Scan(&l.ID, &l.Timestamp, &l.Host, &l.Country, &l.Service, &l.Message, &l.AnomalyScore, &l.IsThreat); err != nil {
			return nil, err
		}
		logs = append(logs, l)
	}
	return logs, nil
}

// GetStats calculates the dashboard counters
func (r *Repository) GetStats() (models.Stats, error) {
	// Simple aggregation for the last 24 hours
	query := `
		SELECT
			count() as total,
			sum(is_threat) as threats
		FROM sentry.logs
		WHERE timestamp > now() - INTERVAL 24 HOUR
	`

	var stats models.Stats
	err := r.db.QueryRow(query).Scan(&stats.TotalLogs, &stats.ThreatCount)
	if err != nil {
		return stats, err
	}

	// EPS Calculation (Approximation based on last minute)
	// In production, use a Materialized View for this
	epsQuery := `SELECT count() / 60 FROM sentry.logs WHERE timestamp > now() - INTERVAL 1 MINUTE`
	_ = r.db.QueryRow(epsQuery).Scan(&stats.EventsPerSec)

	return stats, nil
}