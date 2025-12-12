/**
 * Represents a log entry received from WebSocket or API.
 * Must match the JSON structure from blackbox-tower.
 */
export interface LogEntry {
    id: string;
    timestamp: string; // ISO 8601 string
    host: string;
    country: string;
    service: string;
    message: string;
    anomaly_score: number; // 0.0 to 1.0
    is_threat: number;     // 0 or 1
}

/**
 * Aggregated stats for the top bar.
 */
export interface DashboardStats {
    total_logs: number;
    threat_count: number;
    eps: number;
}

/**
 * Authentication response.
 */
export interface UserSession {
    token: string;
    username: string;
    expiresAt: number;
}