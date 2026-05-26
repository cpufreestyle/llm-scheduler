package backend

import "io"

// Backend 统一接口
type Backend interface {
	ListModels() ([]ModelInfo, error)
	Chat(ChatRequest) (*ChatResponse, error)
	ChatStream(ChatRequest) (io.ReadCloser, error)
	LoadModel(model string) error
	UnloadModel(model string) error
	IsRunning(model string) (bool, error)
	GetModelSize(model string) (int64, error)
	Name() string
}

// BackendManager 多后端管理
type BackendManager struct {
	backends map[string]Backend
	order    []string // 按优先级排序
}

func NewBackendManager() *BackendManager {
	return &BackendManager{
		backends: make(map[string]Backend),
		order:    []string{},
	}
}

func (bm *BackendManager) Register(name string, backend Backend) {
	bm.backends[name] = backend
	bm.order = append(bm.order, name)
}

func (bm *BackendManager) Get(name string) Backend {
	return bm.backends[name]
}

func (bm *BackendManager) Primary() Backend {
	if len(bm.order) > 0 {
		return bm.backends[bm.order[0]]
	}
	return nil
}

func (bm *BackendManager) All() []Backend {
	var all []Backend
	for _, name := range bm.order {
		all = append(all, bm.backends[name])
	}
	return all
}

func (bm *BackendManager) ListAllModels() (map[string][]ModelInfo, error) {
	result := make(map[string][]ModelInfo)
	for name, backend := range bm.backends {
		models, err := backend.ListModels()
		if err == nil {
			result[name] = models
		}
	}
	return result, nil
}
