package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"llm-scheduler/pkg/backend"
	"llm-scheduler/pkg/gpu"
	"llm-scheduler/pkg/scheduler"
	"llm-scheduler/pkg/vram"
)

var sch *scheduler.Scheduler

func main() {
	// Initialize components
	monitor := gpu.NewMonitor()
	monitor.Start()
	
	planner := vram.NewPlanner()
	planner.RegisterGPU("nvidia-5070ti", "nvidia", 16384, 14000)
	planner.RegisterGPU("amd-7900xtx", "amd", 24576, 22000)
	
	ollama := backend.NewOllamaBackend("http://localhost:11434")
	
	sch = scheduler.NewScheduler(scheduler.Config{
		QueueSize:        100,
		BatchTimeoutMs:   100,
		MaxBatchSize:     8,
		ModelUnloadAfter: 300,
	}, monitor, planner, ollama)
	
	sch.Start()
	
	// Setup Gin
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	
	// OpenAI-compatible API
	r.POST("/v1/chat/completions", handleChatCompletion)
	r.GET("/v1/models", handleListModels)
	r.GET("/v1/models/:model", handleGetModel)
	
	// Management API
	r.GET("/api/status", handleStatus)
	r.GET("/api/gpus", handleGPUs)
	r.GET("/api/tasks", handleTasks)
	r.GET("/api/tasks/:id", handleGetTask)
	r.POST("/api/tasks/:id/cancel", handleCancelTask)
	
	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})
	
	port := 8082
	log.Printf("LLM Scheduler running on :%d", port)
	log.Fatal(r.Run(fmt.Sprintf(":%d", port)))
}

// OpenAI-compatible types
type OpenAIRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int          `json:"index"`
	Message      *OpenAIMessage `json:"message,omitempty"`
	Delta        *OpenAIMessage `json:"delta,omitempty"`
	FinishReason string       `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func handleChatCompletion(c *gin.Context) {
	var req OpenAIRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	// Convert to Ollama format
	messages := make([]backend.ChatMessage, len(req.Messages))
	for i, m := range req.Messages {
		messages[i] = backend.ChatMessage{Role: m.Role, Content: m.Content}
	}
	
	chatReq := backend.ChatRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   req.Stream,
	}
	
	taskID := fmt.Sprintf("%d", time.Now().UnixNano())
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	
	schedulerReq := &scheduler.Request{
		ID:         taskID,
		Model:      req.Model,
		Priority:   1,
		Stream:     req.Stream,
		ChatReq:    chatReq,
		ResponseCh: make(chan *scheduler.Response, 100),
		Ctx:        ctx,
		Cancel:     cancel,
	}
	
	if err := sch.Submit(schedulerReq); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	if req.Stream {
		handleStreamingResponse(c, schedulerReq, req.Model, taskID)
	} else {
		handleNormalResponse(c, schedulerReq, req.Model, taskID)
	}
}

func handleNormalResponse(c *gin.Context, req *scheduler.Request, model, taskID string) {
	var content string
	for resp := range req.ResponseCh {
		if resp.Error != nil {
			c.JSON(500, gin.H{"error": resp.Error.Error()})
			return
		}
		content += resp.Content
		if resp.Done {
			break
		}
	}
	
	c.JSON(200, OpenAIResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", taskID),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{{
			Index: 0,
			Message: &OpenAIMessage{
				Role:    "assistant",
				Content: content,
			},
			FinishReason: "stop",
		}},
	})
}

func handleStreamingResponse(c *gin.Context, req *scheduler.Request, model, taskID string) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(500, gin.H{"error": "streaming not supported"})
		return
	}
	
	for resp := range req.ResponseCh {
		if resp.Error != nil {
			data, _ := json.Marshal(map[string]interface{}{"error": resp.Error.Error()})
			fmt.Fprintf(c.Writer, "data: %s\n\n", data)
			flusher.Flush()
			break
		}
		
		chunk := OpenAIResponse{
			ID:      fmt.Sprintf("chatcmpl-%s", taskID),
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []Choice{{
				Index: 0,
				Delta: &OpenAIMessage{
					Role:    "assistant",
					Content: resp.Content,
				},
			}},
		}
		if resp.Done {
			chunk.Choices[0].FinishReason = "stop"
		}
		
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(c.Writer, "data: %s\n\n", data)
		flusher.Flush()
		
		if resp.Done {
			break
		}
	}
	
	fmt.Fprintf(c.Writer, "data: [DONE]\n\n")
	flusher.Flush()
}

func handleListModels(c *gin.Context) {
	c.JSON(200, gin.H{
		"object": "list",
		"data": []gin.H{
			{"id": "qwen2.5:7b", "object": "model", "owned_by": "qwen"},
			{"id": "llama3.1:8b", "object": "model", "owned_by": "meta"},
			{"id": "deepseek-r1:7b", "object": "model", "owned_by": "deepseek"},
		},
	})
}

func handleGetModel(c *gin.Context) {
	model := c.Param("model")
	c.JSON(200, gin.H{
		"id":       model,
		"object":   "model",
		"owned_by": "custom",
	})
}

func handleStatus(c *gin.Context) {
	c.JSON(200, sch.GetStatus())
}

func handleGPUs(c *gin.Context) {
	c.JSON(200, gin.H{"gpus": sch.GetStatus()["gpus"]})
}

func handleTasks(c *gin.Context) {
	c.JSON(200, gin.H{"status": sch.GetStatus()})
}

func handleGetTask(c *gin.Context) {
	id := c.Param("id")
	task := sch.GetTask(id)
	if task == nil {
		c.JSON(404, gin.H{"error": "task not found"})
		return
	}
	c.JSON(200, task)
}

func handleCancelTask(c *gin.Context) {
	id := c.Param("id")
	// Implementation depends on scheduler internals
	c.JSON(200, gin.H{"status": "cancelled", "task_id": id})
}

// Helper function to read streaming response
func readStreamingResponse(body io.Reader, ch chan<- string) {
	defer close(ch)
	decoder := json.NewDecoder(body)
	for {
		var resp map[string]interface{}
		if err := decoder.Decode(&resp); err != nil {
			break
		}
		if msg, ok := resp["message"].(map[string]interface{}); ok {
			if content, ok := msg["content"].(string); ok {
				ch <- content
			}
		}
		if done, ok := resp["done"].(bool); ok && done {
			break
		}
	}
}

func parseModelName(model string) string {
	// Handle model name variations
	return strings.ToLower(strings.TrimSpace(model))
}
