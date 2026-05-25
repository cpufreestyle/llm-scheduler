package gpu

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

type GPUInfo struct {
	ID           string    `json:"id"`
	Type         string    `json:"type"` // nvidia, amd
	VRAMTotalMB  int       `json:"vram_total_mb"`
	VRAMUsedMB   int       `json:"vram_used_mb"`
	VRAMFreeMB   int       `json:"vram_free_mb"`
	Utilization   int       `json:"utilization"` // 0-100
	Temperature   int       `json:"temperature"`
	ModelLoaded  string    `json:"model_loaded"` // 当前加载的模型
	LastUpdate   time.Time `json:"last_update"`
}

type Monitor struct {
	gpus map[string]*GPUInfo
	mu   sync.RWMutex
}

func NewMonitor() *Monitor {
	return &Monitor{
		gpus: make(map[string]*GPUInfo),
	}
}

func (m *Monitor) Start() {
	m.detectGPUs()
	go m.pollLoop()
}

func (m *Monitor) detectGPUs() {
	// NVIDIA GPU
	if nvidiaGPUs := m.detectNVIDIA(); len(nvidiaGPUs) > 0 {
		for _, gpu := range nvidiaGPUs {
			m.gpus[gpu.ID] = gpu
		}
	}
	
	// AMD GPU
	if amdGPUs := m.detectAMD(); len(amdGPUs) > 0 {
		for _, gpu := range amdGPUs {
			m.gpus[gpu.ID] = gpu
		}
	}
}

func (m *Monitor) detectNVIDIA() []*GPUInfo {
	cmd := exec.Command("nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		return nil
	}
	
	var gpus []*GPUInfo
	lines := strings.Split(string(output), "\n")
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 4 {
			continue
		}
		
		vramMB, _ := strconv.Atoi(strings.TrimSpace(parts[2]))
		vramMB = vramMB // nvidia-smi 输出 MB
		
		gpu := &GPUInfo{
			ID:          fmt.Sprintf("nvidia-%d", i),
			Type:        "nvidia",
			VRAMTotalMB: vramMB,
			Utilization: parseInt(strings.TrimSpace(parts[3])),
			Temperature: parseInt(strings.TrimSpace(parts[4])),
			LastUpdate:  time.Now(),
		}
		gpus = append(gpus, gpu)
	}
	return gpus
}

func (m *Monitor) detectAMD() []*GPUInfo {
	// Windows: rocm-smi 不可用，尝试 adl.dll 或 WMI
	// 简化实现：假设已知 7900 XTX 存在
	cmd := exec.Command("wmic", "path", "win32_VideoController", "get", "name,AdapterRAM", "/format:csv")
	output, err := cmd.Output()
	if err != nil {
		return nil
	}
	
	var gpus []*GPUInfo
	re := regexp.MustCompile(`AMD.*?7900`)
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if re.MatchString(line) {
			// 假设 7900 XTX 24GB
			gpu := &GPUInfo{
				ID:          "amd-7900xtx",
				Type:        "amd",
				VRAMTotalMB: 24576,
				Utilization: 0,
				Temperature: 0,
				LastUpdate:  time.Now(),
			}
			gpus = append(gpus, gpu)
		}
	}
	return gpus
}

func (m *Monitor) pollLoop() {
	ticker := time.NewTicker(2 * time.Second)
	for range ticker.C {
		m.updateGPUStats()
	}
}

func (m *Monitor) updateGPUStats() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	// NVIDIA 实时数据
	cmd := exec.Command("nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err == nil {
		lines := strings.Split(string(output), "\n")
		for i, line := range lines {
			if strings.TrimSpace(line) == "" {
				continue
			}
			parts := strings.Split(line, ",")
			if len(parts) < 4 {
				continue
			}
			
			id := fmt.Sprintf("nvidia-%d", i)
			if gpu, ok := m.gpus[id]; ok {
				gpu.VRAMUsedMB = parseInt(strings.TrimSpace(parts[1]))
				gpu.VRAMFreeMB = gpu.VRAMTotalMB - gpu.VRAMUsedMB
				gpu.Utilization = parseInt(strings.TrimSpace(parts[2]))
				gpu.Temperature = parseInt(strings.TrimSpace(parts[3]))
				gpu.LastUpdate = time.Now()
			}
		}
	}
}

func (m *Monitor) GetAllGPUs() []*GPUInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	var result []*GPUInfo
	for _, gpu := range m.gpus {
		result = append(result, gpu)
	}
	return result
}

func (m *Monitor) GetGPU(id string) *GPUInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.gpus[id]
}

func (m *Monitor) SetModelLoaded(gpuID, model string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if gpu, ok := m.gpus[gpuID]; ok {
		gpu.ModelLoaded = model
	}
}

func (m *Monitor) ToJSON() string {
	data, _ := json.MarshalIndent(m.GetAllGPUs(), "", "  ")
	return string(data)
}

func parseInt(s string) int {
	s = strings.TrimSpace(s)
	val, _ := strconv.Atoi(s)
	return val
}
