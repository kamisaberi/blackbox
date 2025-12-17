package database

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"blackbox-vacuum/internal/transport"
	"github.com/redis/go-redis/v9"
)

type RedisCollector struct {
	Addr     string // "localhost:6379"
	Password string
}

func (r *RedisCollector) Name() string {
	return "redis_monitor"
}

func (r *RedisCollector) Start(ctx context.Context, client *transport.CoreClient) {
	rdb := redis.NewClient(&redis.Options{
		Addr:     r.Addr,
		Password: r.Password,
	})

	log.Println("[REDIS] Monitoring...")

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// 1. Check Slow Log (Queries > 10ms usually)
			slowEntries, err := rdb.SlowLogGet(ctx, 10).Result()
			if err == nil {
				for _, entry := range slowEntries {
					// Logic: Avoid resending same ID? 
					// Simple implementation sends recent 10. 
					// In prod, track last ID.
					
					logMap := map[string]interface{}{
						"source":   "redis_slowlog",
						"cmd":      entry.Args,
						"duration": entry.Duration.Microseconds(),
						"client":   entry.ClientAddr,
						"ts":       entry.Time.Unix(),
					}
					
					jsonBytes, _ := json.Marshal(logMap)
					payload := fmt.Sprintf("REDIS: %s\n", string(jsonBytes))
					client.Send([]byte(payload))
				}
			}

			// 2. Check Configuration (Security Audit)
			// Ensure 'requirepass' is set and 'rename-command' is used for FLUSHALL
			config, err := rdb.ConfigGet(ctx, "requirepass").Result()
			if err == nil && len(config) == 2 {
				if config[1] == "" {
					// Alert: No Password Set!
					alert := `{"source":"redis_audit", "alert":"NO_PASSWORD_SET", "severity":"high"}`
					client.Send([]byte(fmt.Sprintf("REDIS_AUDIT: %s\n", alert)))
				}
			}
		}
	}
}