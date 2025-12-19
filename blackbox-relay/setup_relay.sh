#!/bin/bash

# =========================================================
# BLACKBOX RELAY SETUP SCRIPT
# Initializes Go module and installs dependencies
# =========================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}>>> Setting up Blackbox Relay (SOAR)...${NC}"

# 1. Check Go
if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed."
    exit 1
fi

# 2. Initialize Module
if [ ! -f go.mod ]; then
    echo -e "${GREEN}>>> Initializing module: blackbox-relay${NC}"
    go mod init blackbox-relay
else
    echo "go.mod exists. Skipping init."
fi

# 3. Install Dependencies
echo -e "${GREEN}>>> Installing Redis Client...${NC}"
go get github.com/redis/go-redis/v9

# 4. Tidy
echo -e "${GREEN}>>> Tidying dependencies...${NC}"
go mod tidy

echo -e "${BLUE}>>> Setup Complete. Ready to build.${NC}"