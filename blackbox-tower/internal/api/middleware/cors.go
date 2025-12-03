func CorsMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // Allow your laptop's browser to talk to the server
        c.Writer.Header().Set("Access-Control-Allow-Origin", "*") 
        // In production, replace "*" with "https://dashboard.your-company.com"
        
        c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
        c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(204)
            return
        }
        c.Next()
    }
}