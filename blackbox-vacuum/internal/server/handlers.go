package server

import (
	"fmt"
	"io"
	"net/http"
	"strings"

	"blackbox-vacuum/pkg/utils"
)

// Generic Handler: Accepts any JSON, flattens it, sends to Core
func (s *WebhookServer) handleGeneric(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// 1. Sanitize & Format
	// We assume the body is JSON. We minify it to a single line.
	cleanJSON := utils.MinifyJSON(body)
	
	// 2. Wrap for Core
	// Format: "WEBHOOK_GENERIC: {json...}\n"
	payload := fmt.Sprintf("WEBHOOK_GENERIC: %s\n", cleanJSON)

	// 3. Send via TCP
	s.CoreClient.Send([]byte(payload))

	w.WriteHeader(http.StatusOK)
}

// Slack specific handler (Event Subscription)
func (s *WebhookServer) handleSlack(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	
	// Handle Slack URL Verification challenge
	if strings.Contains(string(body), "\"type\":\"url_verification\"") {
		w.Header().Set("Content-Type", "application/json")
		w.Write(body) // Echo back the challenge
		return
	}

	cleanJSON := utils.MinifyJSON(body)
	payload := fmt.Sprintf("SLACK_EVENT: %s\n", cleanJSON)
	
	s.CoreClient.Send([]byte(payload))
	w.WriteHeader(http.StatusOK)
}

// GitHub specific handler
func (s *WebhookServer) handleGitHub(w http.ResponseWriter, r *http.Request) {
	event := r.Header.Get("X-GitHub-Event")
	body, _ := io.ReadAll(r.Body)

	cleanJSON := utils.MinifyJSON(body)
	// Format: GITHUB_PUSH: {...}
	payload := fmt.Sprintf("GITHUB_%s: %s\n", strings.ToUpper(event), cleanJSON)

	s.CoreClient.Send([]byte(payload))
	w.WriteHeader(http.StatusOK)
}