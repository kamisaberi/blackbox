package handlers

import (
	"net/http"
	"strconv"

	"blackbox-tower/internal/storage"

	"github.com/gin-gonic/gin"
)

type QueryHandler struct {
	repo *storage.Repository
}

func NewQueryHandler(repo *storage.Repository) *QueryHandler {
	return &QueryHandler{repo: repo}
}

// GetRecentLogs returns the latest N logs (for the Investigation table)
func (h *QueryHandler) GetRecentLogs(c *gin.Context) {
	limitStr := c.DefaultQuery("limit", "100")
	limit, _ := strconv.Atoi(limitStr)
	if limit > 1000 {
		limit = 1000 // Safety cap
	}

	logs, err := h.repo.GetRecentLogs(limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Database query failed", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, logs)
}

// GetDashboardStats returns aggregated metrics (EPS, Total Threats)
func (h *QueryHandler) GetDashboardStats(c *gin.Context) {
	stats, err := h.repo.GetStats()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to calculate stats", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, stats)
}