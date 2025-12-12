package storage

import (
	"context"
	"log"

	"blackbox-tower/internal/config"

	"github.com/redis/go-redis/v9"
)

type RedisListener struct {
	client *redis.Client
	pubsub *redis.PubSub
}

// NewRedisListener creates a connection to Redis
func NewRedisListener(cfg *config.Config) *RedisListener {
	rdb := redis.NewClient(&redis.Options{
		Addr: cfg.RedisHost + ":" + cfg.RedisPort,
	})

	// Test connection
	_, err := rdb.Ping(context.Background()).Result()
	if err != nil {
		log.Printf("[WARN] Failed to connect to Redis: %v. Real-time alerts will be disabled.", err)
		return nil
	}

	return &RedisListener{client: rdb}
}

// Listen starts a background goroutine to consume alerts.
// It sends every message received into the provided 'broadcast' channel.
func (r *RedisListener) Listen(broadcast chan []byte) {
	if r.client == nil {
		return
	}

	ctx := context.Background()
	// Subscribe to the channel defined in C++ Settings (sentry_alerts)
	r.pubsub = r.client.Subscribe(ctx, "sentry_alerts")

	// Wait for confirmation that subscription is active
	_, err := r.pubsub.Receive(ctx)
	if err != nil {
		log.Printf("[ERR] Failed to subscribe to Redis channel: %v", err)
		return
	}

	log.Println("[INFO] Redis Listener active. Waiting for C++ Core alerts...")

	// Infinite Loop
	ch := r.pubsub.Channel()
	for msg := range ch {
		// msg.Payload contains the JSON string built in pipeline.cpp
		// We forward it directly to the WebSocket hub without parsing it here (Pass-through)
		broadcast <- []byte(msg.Payload)
	}
}

func (r *RedisListener) Close() {
	if r.pubsub != nil {
		r.pubsub.Close()
	}
	if r.client != nil {
		r.client.Close()
	}
}