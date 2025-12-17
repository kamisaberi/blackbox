package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"blackbox-vacuum/internal/collector"
	"blackbox-vacuum/internal/collectors/cloud"
	"blackbox-vacuum/internal/collectors/database"
	"blackbox-vacuum/internal/config"
	"blackbox-vacuum/internal/server"
	"blackbox-vacuum/internal/transport"
)

func main() {
	log.Println("==================================================")
	log.Println("   BLACKBOX VACUUM :: Data Aggregator v0.1.0")
	log.Println("==================================================")

	// 1. Load Configuration (ENV Variables)
	cfg := config.Load()
	log.Printf("[INIT] Core Target: %s:%d", cfg.CoreHost, cfg.CorePort)

	// 2. Initialize TCP Transport (Connection to C++ Core)
	// This handles auto-reconnection logic internally.
	coreClient := transport.NewCoreClient(cfg.CoreHost, cfg.CorePort)
	if err := coreClient.Connect(); err != nil {
		// We don't panic here; we allow the app to start and retry connection later
		log.Printf("[WARN] Initial connection to Blackbox Core failed: %v (Will retry on send)", err)
	} else {
		log.Println("[INIT] Connected to Blackbox Core successfully.")
	}

	// 3. Setup Global Context for Graceful Shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 4. Initialize Collector Registry
	registry := collector.NewRegistry()

	// --- Register Active Collectors based on Config ---

	// A. AWS CloudTrail
	if cfg.AWSRegion != "" {
		log.Printf("[CONF] Enabling AWS Collector (Region: %s)", cfg.AWSRegion)
		registry.Register(&cloud.AWSCollector{Region: cfg.AWSRegion})
	}

	// B. MSSQL Audit
	if cfg.MSSQLConn != "" {
		log.Println("[CONF] Enabling MSSQL Audit Collector")
		registry.Register(&database.MSSQLCollector{ConnString: cfg.MSSQLConn})
	}

	// Postgres
	if cfg.PostgresConn != "" {
    	registry.Register(&database.PostgresCollector{ConnString: cfg.PostgresConn})
	}

	// Okta
	if cfg.OktaDomain != "" {
		registry.Register(&identity.OktaCollector{
			Domain: cfg.OktaDomain, 
			APIToken: cfg.OktaAPIToken,
		})
	}
	// (Add more collectors here as you build them: Okta, Postgres, etc.)

	// 5. Start Polling Loops
	// This launches a goroutine for each registered collector
	registry.StartAll(ctx, coreClient)

	// 6. Start Webhook Server (HTTP Push Listeners)
	webhookSrv := server.NewWebhookServer(cfg.WebhookPort, coreClient)
	
	// Run HTTP server in a goroutine so it doesn't block the signal handler
	go func() {
		log.Println("[INIT] Starting Webhook Server...")
		webhookSrv.Start() // This blocks internally
	}()

	// 7. Block Main Thread until OS Signal
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	log.Println("[INFO] Vacuum running. Waiting for data...")
	<-stop

	// 8. Shutdown Sequence
	log.Println("\n[SHUTDOWN] Signal received. Terminating workers...")
	
	// Cancel context to stop collector loops
	cancel()
	
	// Give workers a moment to finish current HTTP requests/TCP writes
	time.Sleep(1 * time.Second)
	
	log.Println("[SHUTDOWN] Goodbye.")
}