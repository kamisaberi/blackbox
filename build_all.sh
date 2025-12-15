#!/bin/bash

# ==============================================================================
# BLACKBOX MASTER BUILD SCRIPT
# Target: Linux Server (Ubuntu/Debian)
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ANSI Colors for nicer output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
ROOT_DIR=$(pwd)
LOG_FILE="$ROOT_DIR/build.log"

log() {
    echo -e "${BLUE}[BLACKBOX]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# ==============================================================================
# 1. PRE-FLIGHT CHECKS
# ==============================================================================

log "Checking System Environment..."

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    error "This script is designed for Linux (Ubuntu/Debian) only."
fi

# Check for Root/Sudo (Required for installing packages)
if [ "$EUID" -ne 0 ]; then 
    warn "Not running as root. You might be prompted for sudo password during dependency installation."
fi

# ==============================================================================
# 2. INSTALL SYSTEM DEPENDENCIES
# ==============================================================================

install_dependencies() {
    log "Updating package lists and installing dependencies..."
    
    # Common Tools
    sudo apt-get update -y
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        unzip \
        pkg-config

    # C++ Core Dependencies (Boost, Curl, Redis, MaxMind, SSL)
    sudo apt-get install -y \
        libboost-all-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libhiredis-dev \
        libmaxminddb-dev

    # IoT / Sentry Dependencies (Cross compilers optional)
    sudo apt-get install -y \
        gcc-arm-linux-gnueabihf \
        libc6-dev-i386

    # Docker (If not installed)
    if ! command -v docker &> /dev/null; then
        log "Docker not found. Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
        sudo usermod -aG docker $USER || true
        warn "Docker installed. You may need to log out and log back in for group changes to take effect."
    fi

    # Docker Compose
    sudo apt-get install -y docker-compose-plugin || sudo apt-get install -y docker-compose

    # Language Runtimes (Go, Python, Node)
    # We install these to allow native compilation checks, though production uses Docker.
    sudo apt-get install -y \
        golang \
        python3 \
        python3-pip \
        python3-venv \
        nodejs \
        npm

    success "System dependencies installed."
}

install_dependencies

# ==============================================================================
# 3. INITIALIZE DATA DIRECTORIES
# ==============================================================================

log "Initializing Data Directory Structure..."
mkdir -p "$ROOT_DIR/data/config"
mkdir -p "$ROOT_DIR/data/models"
mkdir -p "$ROOT_DIR/data/logs"

# Check for GeoIP
if [ ! -f "$ROOT_DIR/data/config/GeoLite2-City.mmdb" ]; then
    warn "GeoIP Database not found in data/config/."
    warn "Please download GeoLite2-City.mmdb manually (MaxMind license required)."
    # For now, we touch a dummy file so C++ doesn't crash immediately (it handles load failure)
    touch "$ROOT_DIR/data/config/GeoLite2-City.mmdb"
fi

# Create dummy rules if missing
if [ ! -f "$ROOT_DIR/data/config/rules.yaml" ]; then
    touch "$ROOT_DIR/data/config/rules.yaml"
fi

success "Directories ready."

# ==============================================================================
# 4. MODULE: BLACKBOX SIM (The Brain)
# ==============================================================================

build_sim() {
    log "Building Blackbox Sim (Python AI Lab)..."
    cd "$ROOT_DIR/blackbox-sim"
    
    # We build the Docker image because Python environments are messy
    # and we need CUDA support for PyTorch.
    docker build -t blackbox-sim . >> "$LOG_FILE" 2>&1
    
    success "Blackbox Sim built (Docker)."
    
    log "Running Sim to generate artifacts (Vocab/Scaler)..."
    # This generates vocab.txt and scaler_params.txt needed by the Core
    docker run --rm -v "$ROOT_DIR/data/config:/app/data/artifacts" blackbox-sim >> "$LOG_FILE" 2>&1
    
    success "AI Artifacts generated in data/config/."
}

build_sim

# ==============================================================================
# 5. MODULE: BLACKBOX CORE (The Engine)
# ==============================================================================

build_core() {
    log "Building Blackbox Core (C++ Engine)..."
    cd "$ROOT_DIR/blackbox-core"

    # CRITICAL CHECK: CUDA
    if ! command -v nvcc &> /dev/null; then
        warn "NVIDIA CUDA Toolkit (nvcc) not found on host."
        warn "Skipping NATIVE C++ compilation. Building DOCKER image instead."
        
        # Build Docker Image (This compiles C++ inside the container)
        docker build -f docker/Dockerfile -t blackbox-core . >> "$LOG_FILE" 2>&1
    else
        log "CUDA found. Attempting Native Build..."
        mkdir -p build && cd build
        cmake .. >> "$LOG_FILE" 2>&1
        make -j$(nproc) >> "$LOG_FILE" 2>&1
        success "Native binary 'flight-recorder' compiled in blackbox-core/build/"
        
        # Also build Docker image for deployment consistency
        cd "$ROOT_DIR/blackbox-core"
        docker build -f docker/Dockerfile -t blackbox-core . >> "$LOG_FILE" 2>&1
    fi

    success "Blackbox Core built (Docker)."
}

build_core

# ==============================================================================
# 6. MODULE: BLACKBOX TOWER (The API)
# ==============================================================================

build_tower() {
    log "Building Blackbox Tower (Go API)..."
    cd "$ROOT_DIR/blackbox-tower"

    # Native Build check
    if command -v go &> /dev/null; then
        go mod tidy >> "$LOG_FILE" 2>&1
        go build -o atc-server ./cmd/atc-server
        success "Native binary 'atc-server' compiled."
    fi

    # Docker Build
    docker build -t blackbox-tower . >> "$LOG_FILE" 2>&1
    success "Blackbox Tower built (Docker)."
}

build_tower

# ==============================================================================
# 7. MODULE: BLACKBOX HUD (The Frontend)
# ==============================================================================

build_hud() {
    log "Building Blackbox HUD (React Frontend)..."
    cd "$ROOT_DIR/blackbox-hud"

    # Native Build Check
    if command -v npm &> /dev/null; then
        log "Installing NPM dependencies (this might take a minute)..."
        npm install >> "$LOG_FILE" 2>&1
        npm run build >> "$LOG_FILE" 2>&1
        success "Frontend compiled to 'dist/' folder."
    fi

    # Docker Build
    docker build -t blackbox-hud . >> "$LOG_FILE" 2>&1
    success "Blackbox HUD built (Docker)."
}

build_hud

# ==============================================================================
# 8. MODULE: SENTRY MICRO (IoT Agent)
# ==============================================================================

build_sentry() {
    log "Building Blackbox Sentry Micro (IoT Agent)..."
    cd "$ROOT_DIR/blackbox-sentry-micro"

    mkdir -p build && cd build
    cmake .. >> "$LOG_FILE" 2>&1
    make >> "$LOG_FILE" 2>&1
    
    if [ -f "./sentry-micro" ]; then
        success "IoT Agent 'sentry-micro' compiled successfully."
    else
        error "Failed to compile sentry-micro."
    fi
}

build_sentry

# ==============================================================================
# 9. MODULE: TESTS
# ==============================================================================

run_tests() {
    log "Compiling and Running Unit Tests..."
    cd "$ROOT_DIR/blackbox-tests/core"
    
    mkdir -p build && cd build
    cmake .. >> "$LOG_FILE" 2>&1
    make >> "$LOG_FILE" 2>&1
    
    log "Executing Core Logic Tests..."
    ./run_core_tests
    
    success "Core Tests Passed."
}

run_tests

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

echo ""
echo "========================================================"
echo -e "${GREEN}   BUILD COMPLETE ${NC}"
echo "========================================================"
echo "1. Core Engine:   Docker Image (blackbox-core:latest)"
echo "2. Tower API:     Docker Image (blackbox-tower:latest)"
echo "3. HUD:           Docker Image (blackbox-hud:latest)"
echo "4. IoT Agent:     Binary located in blackbox-sentry-micro/build/sentry-micro"
echo "5. Artifacts:     Generated in data/config/"
echo ""
echo "To launch the system:"
echo -e "${BLUE}   cd blackbox-deploy && make up${NC}"
echo ""
echo "Detailed build logs available at: $LOG_FILE"
echo "========================================================"