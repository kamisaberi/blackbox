#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}>>> Setting up Blackbox Vacuum Environment...${NC}"

# 1. Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed. Please install Go first."
    exit 1
fi

# 2. Initialize Go Module (only if go.mod doesn't exist)
MODULE_NAME="blackbox-vacuum"

if [ ! -f go.mod ]; then
    echo -e "${GREEN}>>> Initializing module: $MODULE_NAME${NC}"
    go mod init $MODULE_NAME
else
    echo "go.mod already exists. Skipping init."
fi

# 3. Install Dependencies
echo -e "${GREEN}>>> Installing Core Networking (Gin)...${NC}"
go get github.com/gin-gonic/gin

echo -e "${GREEN}>>> Installing AWS SDK...${NC}"
go get github.com/aws/aws-sdk-go-v2
go get github.com/aws/aws-sdk-go-v2/config
go get github.com/aws/aws-sdk-go-v2/service/cloudtrail

echo -e "${GREEN}>>> Installing SQL Driver (MSSQL)...${NC}"
go get github.com/denisenkom/go-mssqldb

echo -e "${GREEN}>>> Installing Utilities (DotEnv)...${NC}"
go get github.com/joho/godotenv

# 4. Tidy up
echo -e "${GREEN}>>> Tidying up dependencies...${NC}"
# This removes unused dependencies and updates go.sum ensures everything matches
go mod tidy

echo -e "${BLUE}>>> Setup Complete! You are ready to build.${NC}"