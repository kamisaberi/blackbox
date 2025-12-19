package main

import (
	"blackbox-relay/internal/config"
	"blackbox-relay/internal/queue"
	"log"
)

func main() {
	log.Println(">>> Blackbox Relay (SOAR) Starting...")
	
	cfg := config.Load()
	
	// Start blocking consumer loop
	queue.StartConsumer(cfg)
}