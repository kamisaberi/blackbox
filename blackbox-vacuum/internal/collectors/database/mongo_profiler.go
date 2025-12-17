package database

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"blackbox-vacuum/internal/transport"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type MongoCollector struct {
	URI string // "mongodb://user:pass@localhost:27017"
}

func (m *MongoCollector) Name() string {
	return "mongo_profiler"
}

func (m *MongoCollector) Start(ctx context.Context, client *transport.CoreClient) {
	log.Println("[MONGO] Connecting...")
	
	opts := options.Client().ApplyURI(m.URI)
	mongoClient, err := mongo.Connect(ctx, opts)
	if err != nil {
		log.Printf("[MONGO] Connect Error: %v", err)
		return
	}
	defer mongoClient.Disconnect(ctx)

	db := mongoClient.Database("admin") // Profiler is usually in admin or specific DB
	// Note: 'system.profile' must be enabled via db.setProfilingLevel(1) or (2)

	lastTime := time.Now()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Query system.profile
			coll := db.Collection("system.profile")
			
			filter := bson.M{
				"ts": bson.M{"$gt": lastTime},
			}

			cursor, err := coll.Find(ctx, filter)
			if err != nil {
				// Often fails if profiling isn't enabled
				continue 
			}

			var results []bson.M
			if err = cursor.All(ctx, &results); err != nil {
				continue
			}

			for _, doc := range results {
				// Update cursor
				if ts, ok := doc["ts"].(time.Time); ok {
					lastTime = ts
				}

				// Convert BSON to JSON
				jsonBytes, _ := json.Marshal(doc)
				payload := fmt.Sprintf("MONGO: %s\n", string(jsonBytes))
				client.Send([]byte(payload))
			}
		}
	}
}