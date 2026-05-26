#!/bin/bash
# Build script for llm-scheduler binaries

VERSION=${1:-"1.2.0"}
OUTPUT_DIR="./dist"

echo "Building llm-scheduler v${VERSION}..."

mkdir -p $OUTPUT_DIR

# Windows AMD64
echo "Building Windows AMD64..."
GOOS=windows GOARCH=amd64 go build -ldflags="-s -w -X main.Version=${VERSION}" -o $OUTPUT_DIR/llm-scheduler-windows-amd64.exe ./cmd

# Linux AMD64
echo "Building Linux AMD64..."
GOOS=linux GOARCH=amd64 go build -ldflags="-s -w -X main.Version=${VERSION}" -o $OUTPUT_DIR/llm-scheduler-linux-amd64 ./cmd

# Linux ARM64 (for ARM servers)
echo "Building Linux ARM64..."
GOOS=linux GOARCH=arm64 go build -ldflags="-s -w -X main.Version=${VERSION}" -o $OUTPUT_DIR/llm-scheduler-linux-arm64 ./cmd

# macOS AMD64
echo "Building macOS AMD64..."
GOOS=darwin GOARCH=amd64 go build -ldflags="-s -w -X main.Version=${VERSION}" -o $OUTPUT_DIR/llm-scheduler-darwin-amd64 ./cmd

# macOS ARM64 (Apple Silicon)
echo "Building macOS ARM64..."
GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w -X main.Version=${VERSION}" -o $OUTPUT_DIR/llm-scheduler-darwin-arm64 ./cmd

echo "Done! Binaries in $OUTPUT_DIR:"
ls -lh $OUTPUT_DIR/