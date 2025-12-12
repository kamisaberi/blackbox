package websocket

import "log"

// Hub maintains the set of active clients and broadcasts messages to the
type Hub struct {
	// Registered clients.
	clients map[*Client]bool

	// Inbound messages from Redis.
	broadcast chan []byte

	// Register requests from the clients.
	register chan *Client

	// Unregister requests from clients.
	unregister chan *Client
}

func NewHub() *Hub {
	return &Hub{
		broadcast:  make(chan []byte),
		register:   make(chan *Client),
		unregister: make(chan *Client),
		clients:    make(map[*Client]bool),
	}
}

func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.clients[client] = true
			log.Printf("[WS] Client connected. Total: %d", len(h.clients))

		case client := <-h.unregister:
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
				log.Printf("[WS] Client disconnected. Total: %d", len(h.clients))
			}

		case message := <-h.broadcast:
			// A wild alert appeared! Send it to everyone.
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					// If client's buffer is full (slow connection), drop them
					close(client.send)
					delete(h.clients, client)
				}
			}
		}
	}
}

// GetBroadcastChannel allows other modules to push messages into the hub
func (h *Hub) GetBroadcastChannel() chan []byte {
	return h.broadcast
}