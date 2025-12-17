package collector

import (
	"blackbox-vacuum/internal/transport"
	"context"
)

// Collector is the contract for any data fetcher
type Collector interface {
	// Name returns the unique ID (e.g., "aws_cloudtrail")
	Name() string

	// Start begins the polling loop. It runs in a goroutine.
	// client: Used to send data to the C++ Core
	Start(ctx context.Context, client *transport.CoreClient)
}