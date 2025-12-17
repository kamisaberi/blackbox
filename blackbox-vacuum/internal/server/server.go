package server

import (
	"log"
	"net/http"
	"time"

	"blackbox-vacuum/internal/transport"
)

type WebhookServer struct {
	Port       string
	CoreClient *transport.CoreClient
}

func NewWebhookServer(port string, client *transport.CoreClient) *WebhookServer {
	return &WebhookServer{
		Port:       port,
		CoreClient: client,
	}
}

func (s *WebhookServer) Start() {
	mux := http.NewServeMux()

	// Register Routes
	mux.HandleFunc("/hooks/slack", s.handleSlack)
	mux.HandleFunc("/hooks/github", s.handleGitHub)
	mux.HandleFunc("/hooks/generic", s.handleGeneric)

	srv := &http.Server{
		Addr:         ":" + s.Port,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
	}

	log.Printf("[VACUUM] Webhook Server listening on port %s", s.Port)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Printf("[ERR] Webhook server failed: %v", err)
	}
}