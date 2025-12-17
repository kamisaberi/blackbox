package identity

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"blackbox-vacuum/internal/transport"
)

type OktaCollector struct {
	Domain   string // "yourcompany.okta.com"
	APIToken string
}

func (o *OktaCollector) Name() string {
	return "okta_system_log"
}

func (o *OktaCollector) Start(ctx context.Context, client *transport.CoreClient) {
	if o.Domain == "" || o.APIToken == "" {
		return
	}

	log.Println("[OKTA] Poller started...")
	
	// Start polling from "Now"
	// Okta API uses ISO8601
	lastTime := time.Now().UTC().Format("2006-01-02T15:04:05Z")

	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	clientHttp := &http.Client{Timeout: 10 * time.Second}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Construct URL: /api/v1/logs?since=...
			url := fmt.Sprintf("https://%s/api/v1/logs?since=%s&sortOrder=ASCENDING&limit=100", o.Domain, lastTime)
			
			req, _ := http.NewRequest("GET", url, nil)
			req.Header.Add("Authorization", "SSWS "+o.APIToken)
			req.Header.Add("Accept", "application/json")

			resp, err := clientHttp.Do(req)
			if err != nil {
				log.Printf("[OKTA] Request Error: %v", err)
				continue
			}

			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()

			if resp.StatusCode != 200 {
				log.Printf("[OKTA] API returned %d", resp.StatusCode)
				continue
			}

			// Parse Array of Events
			var events []map[string]interface{}
			if err := json.Unmarshal(body, &events); err != nil {
				log.Printf("[OKTA] JSON Error: %v", err)
				continue
			}

			for _, event := range events {
				// Update cursor
				if published, ok := event["published"].(string); ok {
					lastTime = published
				}

				// Extract key fields for AI
				flatLog := map[string]interface{}{
					"source":    "okta",
					"actor":     event["actor"],
					"client":    event["client"],
					"eventType": event["eventType"],
					"outcome":   event["outcome"],
					"ts":        event["published"],
				}

				jsonBytes, _ := json.Marshal(flatLog)
				payload := fmt.Sprintf("OKTA: %s\n", string(jsonBytes))
				client.Send([]byte(payload))
			}
		}
	}
}