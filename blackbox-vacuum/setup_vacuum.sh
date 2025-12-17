#!/bin/bash

# =========================================================
# BLACKBOX VACUUM SETUP SCRIPT (Linux/Mac)
# Installs all dependencies for Cloud, DB, and IoT collectors
# =========================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}>>> Setting up Blackbox Vacuum Environment...${NC}"

# 1. Check if Go is installed
if ! command -v go &> /dev/null; then
    echo -e "${RED}[ERROR] Go is not installed. Please install Go v1.21+ first.${NC}"
    exit 1
fi

# 2. Initialize Go Module (if missing)
MODULE_NAME="blackbox-vacuum"

if [ ! -f go.mod ]; then
    echo -e "${GREEN}>>> Initializing module: $MODULE_NAME${NC}"
    go mod init $MODULE_NAME
else
    echo "go.mod already exists. Skipping init."
fi

# 3. Install Dependencies
echo -e "${GREEN}>>> Installing Core & Web Framework...${NC}"
go get github.com/gin-gonic/gin
go get github.com/joho/godotenv

echo -e "${GREEN}>>> Installing AWS SDK...${NC}"
go get github.com/aws/aws-sdk-go-v2
go get github.com/aws/aws-sdk-go-v2/config
go get github.com/aws/aws-sdk-go-v2/service/cloudtrail

echo -e "${GREEN}>>> Installing Azure SDK...${NC}"
go get github.com/Azure/azure-sdk-for-go/sdk/azidentity
go get github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/monitor/armmonitor

echo -e "${GREEN}>>> Installing Kubernetes Client...${NC}"
go get k8s.io/client-go@latest
go get k8s.io/api@latest
go get k8s.io/apimachinery@latest

echo -e "${GREEN}>>> Installing Database Drivers...${NC}"
go get github.com/denisenkom/go-mssqldb       # SQL Server
go get github.com/lib/pq                      # PostgreSQL
go get go.mongodb.org/mongo-driver/mongo      # MongoDB
go get github.com/redis/go-redis/v9           # Redis

echo -e "${GREEN}>>> Installing SaaS SDKs...${NC}"
go get github.com/slack-go/slack              # Slack

# 4. Cleanup
echo -e "${GREEN}>>> Tidying up dependencies...${NC}"
go mod tidy

echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}   SETUP COMPLETE. READY TO BUILD.${NC}"
echo -e "${BLUE}=====================================================${NC}"