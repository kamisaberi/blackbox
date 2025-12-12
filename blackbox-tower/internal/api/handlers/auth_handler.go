package handlers

import (
	"net/http"
	"time"

	"blackbox-tower/internal/config"
	"blackbox-tower/pkg/models"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

type AuthHandler struct {
	cfg *config.Config
}

func NewAuthHandler(cfg *config.Config) *AuthHandler {
	return &AuthHandler{cfg: cfg}
}

// Login verifies credentials and returns a JWT
func (h *AuthHandler) Login(c *gin.Context) {
	var req models.LoginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload"})
		return
	}

	// 1. Verify Credentials (Hardcoded for MVP)
	// TODO: Replace with DB lookup
	if req.Username != "admin" || req.Password != "blackbox" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid credentials"})
		return
	}

	// 2. Generate JWT Token
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"sub": req.Username,
		"iss": "blackbox-tower",
		"exp": time.Now().Add(24 * time.Hour).Unix(), // 1 Day Expiration
		"role": "admin",
	})

	tokenString, err := token.SignedString([]byte(h.cfg.JWTSecret))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to sign token"})
		return
	}

	// 3. Return Response
	c.JSON(http.StatusOK, models.AuthResponse{
		Token:     tokenString,
		ExpiresAt: time.Now().Add(24 * time.Hour).Unix(),
	})
}