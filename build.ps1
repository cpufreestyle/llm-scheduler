# Build script for llm-scheduler binaries (PowerShell)

param([string]$Version = "1.2.0")

$OutputDir = ".\dist"
$Env:GOPROXY = "https://goproxy.cn,direct"

Write-Host "Building llm-scheduler v$Version..." -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Windows AMD64
Write-Host "Building Windows AMD64..." -ForegroundColor Yellow
$Env:GOOS = "windows"
$Env:GOARCH = "amd64"
go build -ldflags="-s -w" -o "$OutputDir\llm-scheduler-windows-amd64.exe" .\cmd

# Linux AMD64
Write-Host "Building Linux AMD64..." -ForegroundColor Yellow
$Env:GOOS = "linux"
$Env:GOARCH = "amd64"
go build -ldflags="-s -w" -o "$OutputDir\llm-scheduler-linux-amd64" .\cmd

# Linux ARM64
Write-Host "Building Linux ARM64..." -ForegroundColor Yellow
$Env:GOOS = "linux"
$Env:GOARCH = "arm64"
go build -ldflags="-s -w" -o "$OutputDir\llm-scheduler-linux-arm64" .\cmd

# macOS AMD64
Write-Host "Building macOS AMD64..." -ForegroundColor Yellow
$Env:GOOS = "darwin"
$Env:GOARCH = "amd64"
go build -ldflags="-s -w" -o "$OutputDir\llm-scheduler-darwin-amd64" .\cmd

# macOS ARM64 (Apple Silicon)
Write-Host "Building macOS ARM64..." -ForegroundColor Yellow
$Env:GOOS = "darwin"
$Env:GOARCH = "arm64"
go build -ldflags="-s -w" -o "$OutputDir\llm-scheduler-darwin-arm64" .\cmd

# Reset
$Env:GOOS = ""
$Env:GOARCH = ""

Write-Host "Done! Binaries in ${OutputDir}:" -ForegroundColor Green
Get-ChildItem $OutputDir | ForEach-Object { Write-Host "  $($_.Name) - $([math]::Round($_.Length/1MB, 2)) MB" }
