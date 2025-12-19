package integrations

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type SlackPayload struct {
	Text        string       `json:"text"`
	Attachments []Attachment `json:"attachments"`
}

type Attachment struct {
	Color  string `json:"color"`
	Title  string `json:"title"`
	Text   string `json:"text"`
	Footer string `json:"footer"`
	Ts     int64  `json:"ts"`
}

func SendSlackAlert(webhookURL string, alertData map[string]interface{}) error {
	if webhookURL == "" {
		return nil
	}

	score := alertData["score"].(float64)
	ip := alertData["ip"].(string)
	msg := alertData["msg"].(string)

	// Format based on severity
	color := "#ff9900" // Orange (Warning)
	if score > 0.9 {
		color = "#ff0000" // Red (Critical)
	}

	payload := SlackPayload{
		Text: "🚨 *Blackbox Security Alert*",
		Attachments: []Attachment{
			{
				Color:  color,
				Title:  fmt.Sprintf("Threat Detected from %s", ip),
				Text:   fmt.Sprintf("Score: %.2f\nMessage: %s", score, msg),
				Footer: "Blackbox Relay",
				Ts:     time.Now().Unix(),
			},
		},
	}

	body, _ := json.Marshal(payload)
	resp, err := http.Post(webhookURL, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}