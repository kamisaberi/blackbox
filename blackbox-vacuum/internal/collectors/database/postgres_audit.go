package database

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"blackbox-vacuum/internal/transport"
	
	_ "github.com/lib/pq" // Postgres driver
)

type PostgresCollector struct {
	ConnString string // "postgres://user:pass@localhost:5432/dbname?sslmode=disable"
}

type PgStatActivity struct {
	User      string  `json:"user"`
	Database  string  `json:"db"`
	ClientIP  string  `json:"client_ip"`
	Query     string  `json:"query"`
	Duration  float64 `json:"duration_sec"`
	StartTime string  `json:"start_time"`
}

func (p *PostgresCollector) Name() string {
	return "postgres_audit"
}

func (p *PostgresCollector) Start(ctx context.Context, client *transport.CoreClient) {
	log.Println("[PG] Connecting to PostgreSQL...")
	db, err := sql.Open("postgres", p.ConnString)
	if err != nil {
		log.Printf("[PG] Connect Error: %v", err)
		return
	}
	defer db.Close()

	// Validate connection
	if err := db.Ping(); err != nil {
		log.Printf("[PG] Ping Error: %v", err)
		return
	}

	// Poll every 10 seconds
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.pollStats(db, client)
		}
	}
}

func (p *PostgresCollector) pollStats(db *sql.DB, client *transport.CoreClient) {
	// Query: Find active queries running longer than 1 second
	// Security value: Detects SQLi time-delay attacks or DoS attempts
	query := `
		SELECT 
			usename, datname, client_addr, query, 
			EXTRACT(EPOCH FROM (now() - query_start)) as duration,
			query_start
		FROM pg_stat_activity 
		WHERE state = 'active' 
		  AND (now() - query_start) > interval '1 second'
		  AND query NOT LIKE '%pg_stat_activity%'
	`

	rows, err := db.Query(query)
	if err != nil {
		log.Printf("[PG] Query Failed: %v", err)
		return
	}
	defer rows.Close()

	for rows.Next() {
		var row PgStatActivity
		var clientIP sql.NullString // Handle NULL IPs (local connections)

		if err := rows.Scan(&row.User, &row.Database, &clientIP, &row.Query, &row.Duration, &row.StartTime); err != nil {
			continue
		}
		
		if clientIP.Valid {
			row.ClientIP = clientIP.String
		} else {
			row.ClientIP = "localhost"
		}

		// JSON Serialize
		jsonBytes, _ := json.Marshal(row)

		// Format: "POSTGRES: {json}\n"
		payload := fmt.Sprintf("POSTGRES: %s\n", string(jsonBytes))
		client.Send([]byte(payload))
	}
}