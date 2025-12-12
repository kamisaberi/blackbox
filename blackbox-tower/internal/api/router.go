package api

import (
	"time"

	"blackbox-tower/internal/api/handlers"
	"blackbox-tower/internal/config"
	"blackbox-tower/internal/storage"
	"blackbox-tower/internal/websocket"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func SetupRouter(cfg *config.Config, repo *storage.Repository, hub *websocket.Hub) *gin.Engine {
	// Set release mode for production
	// gin.SetMode(gin.ReleaseMode)

	r := gin.Default()

	// 1. Global Middleware: CORS
	// Essential for allowing the React Frontend (on port 3000) to talk to API (on port 8080)
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"}, // In prod, change to "http://localhost:3000"
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Initialize Handlers
	authHandler := handlers.NewAuthHandler(cfg)
	queryHandler := handlers.NewQueryHandler(repo)

	// 2. Public Routes
	api := r.Group("/api/v1")
	{
		api.POST("/login", authHandler.Login)

		// Health Check
		api.GET("/health", func(c *gin.Context) {
			c.JSON(200, gin.H{"status": "alive"})
		})
	}

	// 3. Protected Routes (TODO: Add JWT Middleware here)
	// For MVP, we leave them open or wrap them later.
	protected := api.Group("/")
	{
		protected.GET("/logs", queryHandler.GetRecentLogs)
		protected.GET("/stats", queryHandler.GetDashboardStats)
	}

	// 4. WebSocket Route (The Real-Time Feed)
	r.GET("/ws", func(c *gin.Context) {
		websocket.ServeWs(hub, c)
	})

	return r
}