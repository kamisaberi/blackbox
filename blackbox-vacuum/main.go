package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"blackbox-vacuum/internal/config"
	"blackbox-vacuum/internal/transport"
	"blackbox-vacuum/internal/collectors/cloud"
	"blackbox-vacuum/internal/collectors/database"
)

func main() {
	log.Println(">>> Starting Blackbox Vacuum...")

	// 1. Load Config
	cfg := config.Load()

	// 2. Connect to C++ Core
	client := transport.NewCoreClient(cfg.CoreHost, cfg.CorePort)
	if err := client.Connect(); err != nil {
		log.Printf("[WARN] Core unreachable on startup: %v (Will retry on send)", err)
	}

	// 3. Initialize Collectors
	// In a real app, you'd check config to see which ones are enabled
	collectors := []interface{ Start(context.Context, *transport.CoreClient) }{
		&cloud.AWSCollector{},
		&database.MSSQLCollector{ConnString: cfg.MSSQLConn},
	}

	// 4. Start Loops
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for _, c := range collectors {
		// Go routines for every collector
		// Note: We use reflection or interface grouping in a cleaner way usually
		// Assuming they implement the interface defined above
		go c.Start(ctx, client)
	}

	// 5. Wait for Shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	log.Println(">>> Vacuum shutting down...")
}