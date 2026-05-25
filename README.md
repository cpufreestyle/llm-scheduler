# LLM Scheduler

GPU-aware LLM inference scheduler with OpenAI-compatible API.

## Features

- **GPU-Aware Scheduling**: Automatically routes models to optimal GPU based on VRAM
- **Multi-Backend**: Ollama (primary), vLLM (optional)
- **Request Queue**: Priority-based with batch processing
- **Auto Load/Unload**: Models loaded on demand, unloaded after idle timeout
- **OpenAI Compatible**: Drop-in replacement for OpenAI API

## Architecture

```
Client → OpenAI API → Scheduler → Ollama/vLLM
                          ↓
                     GPU Monitor
                    (VRAM tracking)
```

## GPU Strategy

| Model Size | Preferred GPU | Example |
|------------|---------------|---------|
| ≤7B | NVIDIA 5070Ti (16GB) | qwen2.5:3b, llama3.2:3b |
| 8-15B | NVIDIA 5070Ti | qwen2.5:14b, llama3.1:8b |
| ≥20B | AMD 7900XTX (24GB) | qwen2.5:32b, deepseek-r1:32b |

## Quick Start

```bash
# Build
cd llm-scheduler
go build -o llm-scheduler.exe ./cmd

# Run (requires Ollama running on port 11434)
./llm-scheduler.exe

# Test
curl http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"Hello"}]}'
```

## API Endpoints

### OpenAI Compatible

- `POST /v1/chat/completions` - Chat completion (streaming supported)
- `GET /v1/models` - List models
- `GET /v1/models/:model` - Get model info

### Management

- `GET /api/status` - Scheduler status
- `GET /api/gpus` - GPU status
- `GET /api/tasks` - Task list
- `GET /api/tasks/:id` - Task details
- `POST /api/tasks/:id/cancel` - Cancel task

## Configuration

Edit `config.yaml`:

```yaml
server:
  port: 8082

gpus:
  - id: "nvidia-5070ti"
    type: "nvidia"
    vram_mb: 16384
    max_model_mb: 14000
    
  - id: "amd-7900xtx"
    type: "amd"
    vram_mb: 24576
    max_model_mb: 22000

scheduler:
  queue_size: 100
  batch_timeout_ms: 100
  max_batch_size: 8
  model_unload_after_s: 300
```

## Requirements

- Go 1.22+
- Ollama running on localhost:11434
- NVIDIA GPU: nvidia-smi CLI
- AMD GPU: WMI detection (Windows)

## License

MIT
