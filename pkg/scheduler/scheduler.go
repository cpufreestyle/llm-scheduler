package scheduler

import (
	"container/heap"
	"context"
	"fmt"
	"sync"
	"time"

	"llm-scheduler/pkg/backend"
	"llm-scheduler/pkg/gpu"
	"llm-scheduler/pkg/vram"
)

type Request struct {
	ID         string
	Model      string
	Priority   int       // 0=low, 1=normal, 2=high
	CreatedAt  time.Time
	Stream     bool
	ChatReq    backend.ChatRequest
	ResponseCh chan *Response
	Ctx        context.Context
	Cancel     context.CancelFunc
}

type Response struct {
	Content string
	Done    bool
	Error   error
}

type Task struct {
	ID        string
	Model     string
	GPU       string
	Status    string // pending, running, completed, failed
	CreatedAt time.Time
	StartedAt time.Time
	EndedAt   time.Time
}

type Scheduler struct {
	monitor *gpu.Monitor
	planner *vram.Planner
	ollama  *backend.OllamaBackend
	
	queue    *PriorityQueue
	active   map[string]*Request
	tasks    map[string]*Task
	modelGPU map[string]string // model -> gpu mapping
	
	mu       sync.RWMutex
	stopCh   chan struct{}
	
	config Config
}

type Config struct {
	QueueSize        int
	BatchTimeoutMs   int
	MaxBatchSize     int
	ModelUnloadAfter int // seconds
}

func NewScheduler(cfg Config, monitor *gpu.Monitor, planner *vram.Planner, ollama *backend.OllamaBackend) *Scheduler {
	pq := &PriorityQueue{}
	heap.Init(pq)
	
	return &Scheduler{
		monitor: monitor,
		planner: planner,
		ollama:  ollama,
		queue:   pq,
		active:  make(map[string]*Request),
		tasks:   make(map[string]*Task),
		modelGPU: make(map[string]string),
		stopCh:  make(chan struct{}),
		config:  cfg,
	}
}

func (s *Scheduler) Start() {
	go s.processLoop()
	go s.modelCleanupLoop()
}

func (s *Scheduler) Stop() {
	close(s.stopCh)
}

func (s *Scheduler) Submit(req *Request) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.queue.Len() >= s.config.QueueSize {
		return fmt.Errorf("queue full")
	}
	
	req.CreatedAt = time.Now()
	heap.Push(s.queue, req)
	
	s.tasks[req.ID] = &Task{
		ID:        req.ID,
		Model:     req.Model,
		Status:    "pending",
		CreatedAt: time.Now(),
	}
	
	return nil
}

func (s *Scheduler) processLoop() {
	batch := make([]*Request, 0, s.config.MaxBatchSize)
	timer := time.NewTimer(time.Duration(s.config.BatchTimeoutMs) * time.Millisecond)
	
	for {
		select {
		case <-s.stopCh:
			return
		case <-timer.C:
			if len(batch) > 0 {
				s.processBatch(batch)
				batch = batch[:0]
			}
			timer.Reset(time.Duration(s.config.BatchTimeoutMs) * time.Millisecond)
		default:
			s.mu.Lock()
			if s.queue.Len() > 0 {
				req := heap.Pop(s.queue).(*Request)
				batch = append(batch, req)
			}
			s.mu.Unlock()
			
			if len(batch) >= s.config.MaxBatchSize {
				s.processBatch(batch)
				batch = batch[:0]
				timer.Reset(time.Duration(s.config.BatchTimeoutMs) * time.Millisecond)
			}
		}
	}
}

func (s *Scheduler) processBatch(batch []*Request) {
	// 按模型分组
	byModel := make(map[string][]*Request)
	for _, req := range batch {
		byModel[req.Model] = append(byModel[req.Model], req)
	}
	
	for model, requests := range byModel {
		go s.processModelBatch(model, requests)
	}
}

func (s *Scheduler) processModelBatch(model string, requests []*Request) {
	s.mu.Lock()
	
	// 确保模型已加载
	gpuID, err := s.ensureModelLoaded(model)
	if err != nil {
		s.mu.Unlock()
		for _, req := range requests {
			req.ResponseCh <- &Response{Error: err, Done: true}
		}
		return
	}
	
	// 标记任务为 running
	for _, req := range requests {
		s.active[req.ID] = req
		if task, ok := s.tasks[req.ID]; ok {
			task.Status = "running"
			task.StartedAt = time.Now()
			task.GPU = gpuID
		}
	}
	
	s.mu.Unlock()
	
	// 执行推理
	for _, req := range requests {
		s.executeRequest(req, gpuID)
	}
}

func (s *Scheduler) ensureModelLoaded(model string) (string, error) {
	// 检查是否已加载
	if gpuID, ok := s.modelGPU[model]; ok {
		return gpuID, nil
	}
	
	// 选择 GPU
	gpuID, err := s.planner.SelectBestGPU(model, "")
	if err != nil {
		return "", err
	}
	
	// 加载模型
	if err := s.ollama.LoadModel(model); err != nil {
		return "", fmt.Errorf("failed to load model %s: %w", model, err)
	}
	
	s.modelGPU[model] = gpuID
	s.monitor.SetModelLoaded(gpuID, model)
	
	return gpuID, nil
}

func (s *Scheduler) executeRequest(req *Request, gpuID string) {
	defer func() {
		s.mu.Lock()
		delete(s.active, req.ID)
		if task, ok := s.tasks[req.ID]; ok {
			task.Status = "completed"
			task.EndedAt = time.Now()
		}
		s.mu.Unlock()
	}()
	
	if req.Stream {
		stream, err := s.ollama.ChatStream(req.ChatReq)
		if err != nil {
			req.ResponseCh <- &Response{Error: err, Done: true}
			return
		}
		defer stream.Close()
		
		buf := make([]byte, 4096)
		for {
			n, err := stream.Read(buf)
			if err != nil {
				break
			}
			req.ResponseCh <- &Response{Content: string(buf[:n]), Done: false}
		}
		req.ResponseCh <- &Response{Done: true}
	} else {
		resp, err := s.ollama.Chat(req.ChatReq)
		if err != nil {
			req.ResponseCh <- &Response{Error: err, Done: true}
			return
		}
		req.ResponseCh <- &Response{Content: resp.Message.Content, Done: true}
	}
}

func (s *Scheduler) modelCleanupLoop() {
	ticker := time.NewTicker(30 * time.Second)
	for {
		select {
		case <-s.stopCh:
			return
		case <-ticker.C:
			s.cleanupIdleModels()
		}
	}
}

func (s *Scheduler) cleanupIdleModels() {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if !s.planner.HasGPU() {
		return // CPU 模式下不需要自动卸载
	}
	
	// 检查每个加载的模型是否还有请求
	for model, gpuID := range s.modelGPU {
		hasActive := false
		for _, req := range s.active {
			if req.Model == model {
				hasActive = true
				break
			}
		}
		
		// 如果没有活跃请求且超过阈值，卸载
		if !hasActive {
			s.ollama.UnloadModel(model)
			delete(s.modelGPU, model)
			s.monitor.SetModelLoaded(gpuID, "")
		}
	}
}

func (s *Scheduler) GetStatus() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	return map[string]interface{}{
		"queue_length": s.queue.Len(),
		"active_count": len(s.active),
		"models_loaded": s.modelGPU,
		"gpus":         s.monitor.GetAllGPUs(),
	}
}

func (s *Scheduler) GetTask(id string) *Task {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.tasks[id]
}

// PriorityQueue implementation
type PriorityQueue []*Request

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// Higher priority first, then earlier created
	if pq[i].Priority != pq[j].Priority {
		return pq[i].Priority > pq[j].Priority
	}
	return pq[i].CreatedAt.Before(pq[j].CreatedAt)
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*Request))
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}
