package transport

import (
	"fmt"
	"log"
	"net"
	"sync"
	"time"
)

type CoreClient struct {
	addr string
	conn net.Conn
	mu   sync.Mutex
}

func NewCoreClient(host string, port int) *CoreClient {
	return &CoreClient{
		addr: fmt.Sprintf("%s:%d", host, port),
	}
}

func (c *CoreClient) Connect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	conn, err := net.DialTimeout("tcp", c.addr, 5*time.Second)
	if err != nil {
		return err
	}
	c.conn = conn
	log.Printf("[VACUUM] Connected to Core at %s", c.addr)
	return nil
}

// Send writes data to the TCP socket with auto-reconnect
func (c *CoreClient) Send(data []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn == nil {
		// Try to reconnect silently
		if err := c.reconnect(); err != nil {
			log.Printf("[ERR] Drop packet: Core disconnected")
			return
		}
	}

	_, err := c.conn.Write(data)
	if err != nil {
		log.Printf("[ERR] Write failed: %v. Reconnecting...", err)
		c.conn.Close()
		c.conn = nil
		// Retry logic could go here
	}
}

func (c *CoreClient) reconnect() error {
	conn, err := net.DialTimeout("tcp", c.addr, 2*time.Second)
	if err != nil {
		return err
	}
	c.conn = conn
	return nil
}