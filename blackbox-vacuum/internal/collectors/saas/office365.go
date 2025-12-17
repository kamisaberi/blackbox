package saas

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	"blackbox-vacuum/internal/transport"
)

type O365Collector struct {
	TenantID     string
	ClientID     string
	ClientSecret string
}

func (o *O365Collector) Name() string {
	return "o365_graph"
}

func (o *O365Collector) Start(ctx context.Context, client *transport.CoreClient) {
	log.Println("[O365] Starting Graph API Poller...")

	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			token := o.getAccessToken()
			if token == "" {
				continue
			}
			o.fetchSecurityAlerts(token, client)
		}
	}
}

func (o *O365Collector) getAccessToken() string {
	// Standard OAuth2 Client Credentials Flow
	endpoint := fmt.Sprintf("https://login.microsoftonline.com/%s/oauth2/v2.0/token", o.TenantID)
	data := url.Values{}
	data.Set("client_id", o.ClientID)
	data.Set("scope", "https://graph.microsoft.com/.default")
	data.Set("client_secret", o.ClientSecret)
	data.Set("grant_type", "client_credentials")

	resp, err := http.PostForm(endpoint, data)
	if err != nil {
		log.Printf("[O365] Auth Failed: %v", err)
		return ""
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)
	
	if token, ok := result["access_token"].(string); ok {
		return token
	}
	return ""
}

func (o *O365Collector) fetchSecurityAlerts(token string, client *transport.CoreClient) {
	// Query Microsoft Graph Security API
	req, _ := http.NewRequest("GET", "https://graph.microsoft.com/v1.0/security/alerts?$top=10&$orderby=createdDateTime desc", nil)
	req.Header.Set("Authorization", "Bearer "+token)

	httpClient := &http.Client{Timeout: 10 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("[O365] Request Failed: %v", err)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(body, &result)

	if values, ok := result["value"].([]interface{}); ok {
		for _, v := range values {
			alert := v.(map[string]interface{})
			
			// Format
			logMap := map[string]interface{}{
				"source":      "o365",
				"title":       alert["title"],
				"severity":    alert["severity"],
				"category":    alert["category"],
				"description": alert["description"],
				"ts":          time.Now().Unix(), // or parse alert["createdDateTime"]
			}

			jsonBytes, _ := json.Marshal(logMap)
			payload := fmt.Sprintf("O365: %s\n", string(jsonBytes))
			client.Send([]byte(payload))
		}
	}
}