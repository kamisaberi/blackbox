package collector

import (
	"blackbox-vacuum/internal/transport"
	"context"
	"sync"
)

type Registry struct {
	collectors []Collector
	mu         sync.Mutex
}

func NewRegistry() *Registry {
	return &Registry{
		collectors: make([]Collector, 0),
	}
}

func (r *Registry) Register(c Collector) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.collectors = append(r.collectors, c)
}

// StartAll launches all registered collectors in separate goroutines
func (r *Registry) StartAll(ctx context.Context, client *transport.CoreClient) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, c := range r.collectors {
		go c.Start(ctx, client)
	}
}