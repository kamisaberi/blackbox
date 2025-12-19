#!/bin/bash

# =========================================================
# BLACKBOX ENTERPRISE LAUNCHER
# Builds and starts Core, Tower, HUD, Vacuum, and Relay
# =========================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}>>> Initializing Blackbox Enterprise Stack...${NC}"

# 1. Check Directory Context
if [ ! -d "blackbox-deploy" ]; then
    echo -e "${YELLOW}[ERROR] blackbox-deploy folder not found.${NC}"
    echo "Please run this script from the root 'blackbox' directory."
    exit 1
fi

# 2. Configuration Prompt (Slack)
echo -e "${YELLOW}Optional: Configure Slack Alerts${NC}"
echo "Paste your Slack Webhook URL below (or press Enter to skip):"
read -r SLACK_INPUT

if [ -z "$SLACK_INPUT" ]; then
    echo "No Webhook provided. Slack alerts will be disabled."
    export SLACK_WEBHOOK_URL=""
else
    export SLACK_WEBHOOK_URL="$SLACK_INPUT"
    echo -e "${GREEN}Slack integration enabled.${NC}"
fi

# 3. Docker Compose Launch
echo -e "${BLUE}>>> Building and Starting Services...${NC}"
cd blackbox-deploy

# Using the specific compose file path
docker-compose -f compose/docker-compose.yml up -d --build

# 4. Verification
echo -e "${GREEN}>>> Stack is Online!${NC}"
echo "---------------------------------------------------"
echo "  [HUD]      http://localhost:3000"
echo "  [API]      http://localhost:8080"
echo "  [DB]       http://localhost:8123"
echo "---------------------------------------------------"

echo -e "${BLUE}>>> Checking Relay Connection status...${NC}"
# Wait a moment for container to initialize
sleep 2
docker logs blackbox-relay | grep "Listening" || echo "Relay starting up..."