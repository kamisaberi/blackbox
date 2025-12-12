# Blackbox Master Makefile

.PHONY: all build up down logs clean shell-core

all: build up

# Build all Docker images
build:
	@echo ">>> Building Blackbox Stack..."
	docker-compose -f compose/docker-compose.yml build

# Start the stack (Detached)
up:
	@echo ">>> Starting Kinetic Defense..."
	docker-compose -f compose/docker-compose.yml up -d
	@echo ">>> System Online."
	@echo "    - HUD:   http://localhost:3000"
	@echo "    - Tower: http://localhost:8080"
	@echo "    - Core:  UDP:514 / TCP:601"

# Stop the stack
down:
	@echo ">>> Stopping System..."
	docker-compose -f compose/docker-compose.yml down

# View Logs (Tail)
logs:
	docker-compose -f compose/docker-compose.yml logs -f

# Debug: Shell into Core
shell-core:
	docker exec -it blackbox-core /bin/bash

# Clean Docker artifacts
clean:
	docker-compose -f compose/docker-compose.yml down -v
	docker system prune -f