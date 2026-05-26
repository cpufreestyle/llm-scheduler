package backend

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type OllamaBackend struct {
	BaseURL string
	Client  *http.Client
}

func NewOllamaBackend(baseURL string) *OllamaBackend {
	return &OllamaBackend{
		BaseURL: baseURL,
		Client: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

func (b *OllamaBackend) Name() string { return "ollama" }

type ModelInfo struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
}

type ModelResponse struct {
	Models []ModelInfo `json:"models"`
}

func (b *OllamaBackend) ListModels() ([]ModelInfo, error) {
	resp, err := b.Client.Get(b.BaseURL + "/api/tags")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var modelResp ModelResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelResp); err != nil {
		return nil, err
	}
	return modelResp.Models, nil
}

type ChatRequest struct {
	Model    string                 `json:"model"`
	Messages []ChatMessage          `json:"messages"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResponse struct {
	Model     string      `json:"model"`
	CreatedAt time.Time   `json:"created_at"`
	Message   ChatMessage `json:"message"`
	Done      bool        `json:"done"`
}

func (b *OllamaBackend) Chat(req ChatRequest) (*ChatResponse, error) {
	body, _ := json.Marshal(req)
	httpReq, _ := http.NewRequest("POST", b.BaseURL+"/api/chat", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := b.Client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	// 非流式响应，读取最后一个 done:true 的响应体
	respBody, _ := io.ReadAll(resp.Body)
	
	// Ollama 返回多行 JSON，取最后一行
	lines := bytes.Split(respBody, []byte("\n"))
	for i := len(lines) - 1; i >= 0; i-- {
		if len(lines[i]) == 0 {
			continue
		}
		var chatResp ChatResponse
		if err := json.Unmarshal(lines[i], &chatResp); err == nil && chatResp.Done {
			return &chatResp, nil
		}
	}
	
	return nil, fmt.Errorf("no valid response")
}

func (b *OllamaBackend) ChatStream(req ChatRequest) (io.ReadCloser, error) {
	req.Stream = true
	body, _ := json.Marshal(req)
	httpReq, _ := http.NewRequest("POST", b.BaseURL+"/api/chat", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := b.Client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	return resp.Body, nil
}

type LoadModelRequest struct {
	Model   string `json:"model"`
	KeepAI  bool   `json:"keep_alive,omitempty"`
}

func (b *OllamaBackend) LoadModel(model string) error {
	body, _ := json.Marshal(LoadModelRequest{Model: model, KeepAI: true})
	httpReq, _ := http.NewRequest("POST", b.BaseURL+"/api/generate", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := b.Client.Do(httpReq)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func (b *OllamaBackend) UnloadModel(model string) error {
	body, _ := json.Marshal(map[string]interface{}{
		"model":      model,
		"keep_alive": 0,
	})
	httpReq, _ := http.NewRequest("POST", b.BaseURL+"/api/generate", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := b.Client.Do(httpReq)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func (b *OllamaBackend) IsRunning(model string) (bool, error) {
	resp, err := b.Client.Get(b.BaseURL + "/api/ps")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	
	var psResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&psResp); err != nil {
		return false, err
	}
	
	for _, m := range psResp.Models {
		if m.Name == model {
			return true, nil
		}
	}
	return false, nil
}

func (b *OllamaBackend) GetModelSize(model string) (int64, error) {
	models, err := b.ListModels()
	if err != nil {
		return 0, err
	}
	for _, m := range models {
		if m.Name == model {
			return m.Size, nil
		}
	}
	return 0, fmt.Errorf("model not found: %s", model)
}
