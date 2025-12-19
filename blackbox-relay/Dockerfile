# Build Stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build static binary
RUN CGO_ENABLED=0 GOOS=linux go build -o relay ./cmd/relay

# Runtime Stage
FROM alpine:latest

WORKDIR /app
COPY --from=builder /app/relay .

# Environment variables will be injected by Docker Compose
CMD ["./relay"]