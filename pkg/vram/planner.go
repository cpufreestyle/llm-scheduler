package vram

import (
	"strings"
)

// 模型大小估算（基于参数量）
var ModelVRAMEstimates = map[string]int{
	// Qwen
	"qwen2.5:0.5b":  600,
	"qwen2.5:1.5b":  1500,
	"qwen2.5:3b":   3000,
	"qwen2.5:7b":   6000,
	"qwen2.5:14b":  12000,
	"qwen2.5:32b":  22000,
	"qwen2.5:72b":  48000, // 需要量化
	
	// Llama
	"llama3.2:1b":   1200,
	"llama3.2:3b":   3000,
	"llama3.1:8b":   7000,
	"llama3.3:70b":  42000, // 4-bit 量化
	
	// Gemma
	"gemma2:2b":  2000,
	"gemma2:9b":  8000,
	"gemma2:27b": 24000,
	
	// DeepSeek
	"deepseek-r1:1.5b": 1500,
	"deepseek-r1:7b":   6000,
	"deepseek-r1:8b":   7000,
	"deepseek-r1:14b":   12000,
	"deepseek-r1:32b":   22000,
	"deepseek-r1:70b":   45000,
}

type GPUCapacity struct {
	ID          string
	Type        string
	VRAMTotalMB int
	VRAMFreeMB  int
	MaxModelMB  int // 预留后的最大可用
}

type Planner struct {
	gpus map[string]*GPUCapacity
}

func NewPlanner() *Planner {
	return &Planner{
		gpus: make(map[string]*GPUCapacity),
	}
}

func (p *Planner) RegisterGPU(id, gpuType string, vramTotalMB, maxModelMB int) {
	p.gpus[id] = &GPUCapacity{
		ID:          id,
		Type:        gpuType,
		VRAMTotalMB: vramTotalMB,
		MaxModelMB:  maxModelMB,
		VRAMFreeMB:  vramTotalMB,
	}
}

func (p *Planner) UpdateFreeVRAM(id string, freeMB int) {
	if gpu, ok := p.gpus[id]; ok {
		gpu.VRAMFreeMB = freeMB
	}
}

// EstimateModelVRAM 估算模型所需 VRAM
func (p *Planner) EstimateModelVRAM(model string) int {
	// 精确匹配
	if vram, ok := ModelVRAMEstimates[model]; ok {
		return vram
	}
	
	// 模糊匹配（去掉量化后缀）
	baseModel := model
	for _, suffix := range []string{"-q4", "-q5", "-q8", "-fp16", "-f16"} {
		if strings.Contains(model, suffix) {
			baseModel = strings.Split(model, suffix)[0]
			break
		}
	}
	if vram, ok := ModelVRAMEstimates[baseModel]; ok {
		return vram
	}
	
	// 按参数量估算（模型名含数字）
	// 7B ≈ 6GB (4-bit), 14GB (fp16)
	// 简化：参数量(B) * 2000 = VRAM(MB) (4-bit)
	for name, vram := range ModelVRAMEstimates {
		if strings.Contains(name, model) || strings.Contains(model, name) {
			return vram
		}
	}
	
	// 默认保守估算：10GB
	return 10000
}

// HasGPU 检查是否注册了 GPU
func (p *Planner) HasGPU() bool {
	return len(p.gpus) > 0
}

// SelectBestGPU 选择最适合的 GPU
// 当没有注册 GPU 时返回 "cpu" 表示 CPU 直通模式
func (p *Planner) SelectBestGPU(model string, preferredGPU string) (string, error) {
	// 无 GPU 注册时进入直通模式
	if len(p.gpus) == 0 {
		return "cpu", nil
	}
	
	requiredVRAM := p.EstimateModelVRAM(model)
	
	// 优先使用指定 GPU
	if preferredGPU != "" {
		if gpu, ok := p.gpus[preferredGPU]; ok {
			if gpu.VRAMFreeMB >= requiredVRAM && gpu.MaxModelMB >= requiredVRAM {
				return preferredGPU, nil
			}
		}
	}
	
	// 自动选择：显存足够 + 显存利用率最优
	var bestGPU string
	bestScore := -1
	
	for id, gpu := range p.gpus {
		if gpu.VRAMFreeMB < requiredVRAM || gpu.MaxModelMB < requiredVRAM {
			continue
		}
		
		// 评分：显存刚好够用得分最高（binpack 策略）
		score := 10000 - (gpu.VRAMFreeMB - requiredVRAM)
		if gpu.Type == "nvidia" {
			score += 100 // NVIDIA 优先（CUDA 生态）
		}
		
		if score > bestScore {
			bestScore = score
			bestGPU = id
		}
	}
	
	if bestGPU == "" {
		// 无合适 GPU 也退化为直通模式
		return "cpu", nil
	}
	
	return bestGPU, nil
}

// GetModelCategory 根据模型大小分类
func (p *Planner) GetModelCategory(model string) string {
	vram := p.EstimateModelVRAM(model)
	switch {
	case vram <= 8000:
		return "small"
	case vram <= 14000:
		return "medium"
	default:
		return "large"
	}
}

func (p *Planner) GetGPUStatus() []GPUCapacity {
	var result []GPUCapacity
	for _, gpu := range p.gpus {
		result = append(result, *gpu)
	}
	return result
}
