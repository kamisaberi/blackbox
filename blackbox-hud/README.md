# 👁️ Blackbox HUD
### The Analyst Dashboard & Command Center

[![Stack](https://img.shields.io/badge/stack-React%20%7C%20TypeScript-blue)]()
[![Build](https://img.shields.io/badge/build-Vite-purple)]()
[![Style](https://img.shields.io/badge/style-Tailwind-38bdf8)]()

**Blackbox HUD (Heads-Up Display)** is the visualization layer of the Blackbox platform. It is a single-page application (SPA) built to render high-velocity security telemetry without browser lag.

Unlike standard admin panels, this dashboard is engineered for **Security Operations Centers (SOCs)**. It uses list virtualization to handle infinite scrolling log streams and WebSockets for sub-millisecond alert visualization.

---

## ⚡ Key Capabilities

### 1. "The Matrix" Log Stream
*   **Virtualization:** Uses `react-window` to render only the visible rows of the log table. This allows the browser to hold millions of logs in memory while maintaining 60 FPS scrolling.
*   **Live Mode:** Automatically auto-scrolls as new logs arrive via WebSocket.

### 2. Real-Time Telemetry
*   **Velocity Charts:** Visualizes Events Per Second (EPS) using `recharts` to detect DDoS spikes instantly.
*   **Threat Counters:** Live aggregation of "Critical" vs "Info" logs.

### 3. Investigation Interface
*   **Drill-Down:** Click on an IP to pivot to the Investigation view.
*   **Search:** Filters historical data stored in ClickHouse via the Tower API.

### 4. SOC-Native Design
*   **Dark Mode Only:** Designed for low-light control rooms.
*   **Keyboard Shortcuts:** Optimized for power users.

---

## 🛠️ Build Instructions

### Prerequisites
*   Node.js 18+
*   NPM or Yarn

### 1. Local Development
This runs the frontend in dev mode with Hot Module Replacement (HMR).

```bash
# Install dependencies
npm install

# Start Dev Server
npm run dev
```
*   Access at: `http://localhost:3000`
*   *Note: Ensure `blackbox-tower` is running on port 8080 for API calls.*

### 2. Production Build
Compiles TypeScript and React into static HTML/JS/CSS assets.

```bash
npm run build
# Output is located in /dist
```

### 3. Docker Build
Builds a lightweight Nginx container serving the static assets.

```bash
docker build -t blackbox-hud .
```

---

## ⚙️ Configuration

Configuration is handled via `.env` files (Vite standard).

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VITE_API_URL` | `http://localhost:8080` | URL of the Blackbox Tower REST API. |
| `VITE_WS_URL` | `ws://localhost:8080/ws` | URL of the WebSocket Alert Stream. |

**Development Proxy:**
In `vite.config.ts`, requests to `/api` are automatically proxied to `localhost:8080` to avoid CORS issues during development.

---

## 📂 Project Structure

```text
src/
├── api/               # Axios clients & API definition
├── assets/            # Static images & icons
├── components/
│   ├── common/        # Buttons, Inputs, Cards
│   ├── layout/        # Sidebar, Header
│   ├── stream/        # The Virtualized Log Viewer
│   └── visualizations/# Charts & Graphs
├── hooks/             # Custom React Hooks (useWebSocket)
├── pages/             # Route views (Dashboard, Login, Settings)
├── store/             # Zustand Global State Manager
├── types/             # TypeScript interfaces (LogEntry, Stats)
└── utils/             # Formatters & Helpers
```

---

## 🧪 Tech Stack Details

*   **Framework:** React 18
*   **Language:** TypeScript
*   **Build Tool:** Vite
*   **Styling:** Tailwind CSS
*   **State Management:** Zustand (Chosen for performance over Redux)
*   **Charts:** Recharts
*   **Icons:** Lucide React
*   **Networking:** Axios + Native WebSocket API

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
