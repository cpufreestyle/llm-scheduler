package backend

import (
	"testing"
)

func TestBackendManager_Register(t *testing.T) {
	bm := NewBackendManager()
	
	// Create mock backends
	b1 := &mockBackend{name: "backend1"}
	b2 := &mockBackend{name: "backend2"}
	
	bm.Register("b1", b1)
	bm.Register("b2", b2)
	
	if len(bm.backends) != 2 {
		t.Errorf("Expected 2 backends, got %d", len(bm.backends))
	}
	
	if bm.order[0] != "b1" {
		t.Errorf("Expected first backend to be b1, got %s", bm.order[0])
	}
}

func TestBackendManager_Primary(t *testing.T) {
	bm := NewBackendManager()
	
	// No backends
	if bm.Primary() != nil {
		t.Error("Expected nil when no backends registered")
	}
	
	// Add backend
	bm.Register("primary", &mockBackend{name: "primary"})
	
	if bm.Primary().Name() != "primary" {
		t.Errorf("Expected primary backend, got %s", bm.Primary().Name())
	}
}

func TestBackendManager_Get(t *testing.T) {
	bm := NewBackendManager()
	bm.Register("test", &mockBackend{name: "test"})
	
	b := bm.Get("test")
	if b == nil {
		t.Error("Expected to get backend")
	}
	if b.Name() != "test" {
		t.Errorf("Expected name 'test', got %s", b.Name())
	}
	
	// Non-existent
	if bm.Get("nonexistent") != nil {
		t.Error("Expected nil for non-existent backend")
	}
}

// Mock backend for testing
type mockBackend struct {
	name string
}

func (m *mockBackend) Name() string { return m.name }
func (m *mockBackend) ListModels() ([]ModelInfo, error) { return nil, nil }
func (m *mockBackend) Chat(req ChatRequest) (*ChatResponse, error) { return nil, nil }
func (m *mockBackend) ChatStream(req ChatRequest) (interface{}, error) { return nil, nil }
func (m *mockBackend) LoadModel(model string) error { return nil }
func (m *mockBackend) UnloadModel(model string) error { return nil }
func (m *mockBackend) IsRunning(model string) (bool, error) { return false, nil }
func (m *mockBackend) GetModelSize(model string) (int64, error) { return 0, nil }