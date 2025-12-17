package saas

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"blackbox-vacuum/internal/transport"
	"blackbox-vacuum/pkg/utils"

	"github.com/slack-go/slack"
)

type SlackCollector struct {
	APIToken  string
	ChannelID string // Optional: specific channel to watch
}

func (s *SlackCollector) Name() string {
	return "slack_poller"
}

func (s *SlackCollector) Start(ctx context.Context, client *transport.CoreClient) {
	if s.APIToken == "" {
		return
	}

	api := slack.New(s.APIToken)
	log.Println("[SLACK] Poller started...")

	// Keep track of the last message timestamp to avoid duplicates
	lastTimestamp := ""

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Fetch History
			params := &slack.GetConversationHistoryParameters{
				ChannelID: s.ChannelID, // e.g. "C12345678"
				Limit:     10,
			}
			if lastTimestamp != "" {
				params.Oldest = lastTimestamp
			}

			history, err := api.GetConversationHistory(params)
			if err != nil {
				log.Printf("[SLACK] Error fetching history: %v", err)
				continue
			}

			for i := len(history.Messages) - 1; i >= 0; i-- {
				msg := history.Messages[i]
				lastTimestamp = msg.Timestamp

				// Skip bot messages if desired
				if msg.SubType == "bot_message" {
					continue
				}

				// Sanitize content (remove passwords/keys if someone pasted them)
				cleanText := utils.Sanitize(msg.Text)

				logEntry := map[string]interface{}{
					"source": "slack",
					"user":   msg.User,
					"text":   cleanText,
					"ts":     msg.Timestamp,
				}

				jsonBytes, _ := json.Marshal(logEntry)
				payload := fmt.Sprintf("SLACK_HIST: %s\n", string(jsonBytes))
				client.Send([]byte(payload))
			}
		}
	}
}