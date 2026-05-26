package vram

import (
	"testing"
)

func TestPlanner_EstimateModelVRAM(t *testing.T) {
	p := NewPlanner()
	
	// Known model
	vram := p.EstimateModelVRAM("qwen2.5:7b")
	if vram != 6000 {
		t.Errorf("Expected 6000MB for qwen2.5:7b, got %d", vram)
	}
	
	// Large model
	vram = p.EstimateModelVRAM("qwen2.5:32b")
	if vram != 22000 {
		t.Errorf("Expected 22000MB for qwen2.5:32b, got %d", vram)
	}
	
	// Unknown model (default)
	vram = p.EstimateModelVRAM("unknown-model")
	if vram != 10000 {
		t.Errorf("Expected 10000MB for unknown model, got %d", vram)
	}
}

func TestPlanner_SelectBestGPU(t *testing.T) {
	p := NewPlanner()
	
	// Register GPUs
	p.RegisterGPU("nvidia-5070ti", "nvidia", 16000, 14000)
	p.RegisterGPU("amd-7900xtx", "amd", 24576, 22000)
	
	// Small model should fit on NVIDIA
	gpu, err := p.SelectBestGPU("qwen2.5:3b", "")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if gpu != "nvidia-5070ti" {
		t.Errorf("Expected nvidia-5070ti for small model, got %s", gpu)
	}
	
	// Large model needs AMD
	gpu, err = p.SelectBestGPU("qwen2.5:32b", "")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if gpu != "amd-7900xtx" {
		t.Errorf("Expected amd-7900xtx for large model, got %s", gpu)
	}
	
	// Prefer specific GPU if it can fit
	gpu, err = p.SelectBestGPU("qwen2.5:3b", "amd-7900xtx")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if gpu != "amd-7900xtx" {
		t.Errorf("Expected amd-7900xtx when preferred, got %s", gpu)
	}
}

func TestPlanner_GetModelCategory(t *testing.T) {
	p := NewPlanner()
	
	if p.GetModelCategory("qwen2.5:3b") != "small" {
		t.Errorf("Expected 'small' for 3b model")
	}
	
	if p.GetModelCategory("qwen2.5:14b") != "medium" {
		t.Errorf("Expected 'medium' for 14b model")
	}
	
	if p.GetModelCategory("qwen2.5:32b") != "large" {
		t.Errorf("Expected 'large' for 32b model")
	}
}

func TestPlanner_RegisterGPU(t *testing.T) {
	p := NewPlanner()
	p.RegisterGPU("test-gpu", "nvidia", 16000, 14000)
	
	status := p.GetGPUStatus()
	if len(status) != 1 {
		t.Errorf("Expected 1 GPU, got %d", len(status))
	}
	
	if status[0].ID != "test-gpu" {
		t.Errorf("Expected 'test-gpu', got %s", status[0].ID)
	}
	
	if status[0].VRAMFreeMB != 16000 {
		t.Errorf("Expected initial free VRAM 16000, got %d", status[0].VRAMFreeMB)
	}
}

func TestPlanner_UpdateFreeVRAM(t *testing.T) {
	p := NewPlanner()
	p.RegisterGPU("test-gpu", "nvidia", 16000, 14000)
	
	p.UpdateFreeVRAM("test-gpu", 8000)
	
	status := p.GetGPUStatus()
	if status[0].VRAMFreeMB != 8000 {
		t.Errorf("Expected 8000MB free, got %d", status[0].VRAMFreeMB)
	}
}