package queue

import (
	"context"
	"encoding/json"
	"log"

	"blackbox-relay/internal/config"
	"blackbox-relay/internal/integrations"

	"github.com/redis/go-redis/v9"
)

func StartConsumer(cfg *config.Config) {
	rdb := redis.NewClient(&redis.Options{
		Addr: cfg.RedisHost + ":" + cfg.RedisPort,
	})

	ctx := context.Background()
	pubsub := rdb.Subscribe(ctx, "sentry_alerts")
	defer pubsub.Close()

	log.Println("[RELAY] Listening for alerts on Redis channel: sentry_alerts")

	ch := pubsub.Channel()

	for msg := range ch {
		// 1. Parse JSON from C++ Core
		var alertData map[string]interface{}
		if err := json.Unmarshal([]byte(msg.Payload), &alertData); err != nil {
			log.Printf("[ERR] Invalid JSON: %v", err)
			continue
		}

		log.Printf("[RELAY] Processing Alert: %v", alertData)

		// 2. Dispatch to Slack
		if cfg.SlackWebhookURL != "" {
			go func() {
				if err := integrations.SendSlackAlert(cfg.SlackWebhookURL, alertData); err != nil {
					log.Printf("[ERR] Slack failed: %v", err)
				}
			}()
		}

		// 3. (Future) Dispatch to PagerDuty / Jira here
	}
}