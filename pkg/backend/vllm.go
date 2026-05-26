package backend

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// vLLMBackend implements Backend for vLLM server
type vLLMBackend struct {
	name    string
	baseURL string
	client  *http.Client
	online  bool
	apiKey  string
}

func NewVLLMBackend(baseURL string, apiKey string) *vLLMBackend {
	return &vLLMBackend{
		name:    "vllm",
		baseURL: strings.TrimSuffix(baseURL, "/"),
		client:  &http.Client{Timeout: 5 * time.Minute},
		apiKey:  apiKey,
	}
}

func (b *vLLMBackend) Name() string { return b.name }

func (b *vLLMBackend) ListModels() ([]ModelInfo, error) {
	req, _ := http.NewRequest("GET", b.baseURL+"/v1/models", nil)
	if b.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.apiKey)
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Data []struct {
			ID   string `json:"id"`
			Size int64  `json:"size,omitempty"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	models := make([]ModelInfo, len(result.Data))
	for i, m := range result.Data {
		models[i] = ModelInfo{Name: m.ID, Size: m.Size}
	}
	return models, nil
}

func (b *vLLMBackend) Chat(req ChatRequest) (*ChatResponse, error) {
	// Convert to OpenAI format
	openaiReq := map[string]interface{}{
		"model":    req.Model,
		"messages": req.Messages,
		"stream":   false,
	}
	bodyBytes, _ := json.Marshal(openaiReq)

	httpReq, _ := http.NewRequest("POST", b.baseURL+"/v1/chat/completions", bytes.NewReader(bodyBytes))
	httpReq.Header.Set("Content-Type", "application/json")
	if b.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+b.apiKey)
	}

	resp, err := b.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("vLLM error %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Message ChatMessage `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("no response from vLLM")
	}

	return &ChatResponse{
		Model:     result.Model,
		CreatedAt: time.Now(),
		Message:   result.Choices[0].Message,
		Done:      true,
	}, nil
}

func (b *vLLMBackend) ChatStream(req ChatRequest) (io.ReadCloser, error) {
	openaiReq := map[string]interface{}{
		"model":    req.Model,
		"messages": req.Messages,
		"stream":   true,
	}
	bodyBytes, _ := json.Marshal(openaiReq)

	httpReq, _ := http.NewRequest("POST", b.baseURL+"/v1/chat/completions", bytes.NewReader(bodyBytes))
	httpReq.Header.Set("Content-Type", "application/json")
	if b.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+b.apiKey)
	}

	resp, err := b.client.Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("vLLM error %d: %s", resp.StatusCode, string(body))
	}

	return resp.Body, nil
}

func (b *vLLMBackend) LoadModel(model string) error {
	// vLLM loads models at startup, runtime loading not supported in basic mode
	return fmt.Errorf("vLLM requires restart to load new models")
}

func (b *vLLMBackend) UnloadModel(model string) error {
	// vLLM doesn't support runtime unload
	return fmt.Errorf("vLLM does not support runtime model unload")
}

func (b *vLLMBackend) IsRunning(model string) (bool, error) {
	models, err := b.ListModels()
	if err != nil {
		return false, err
	}
	for _, m := range models {
		if m.Name == model {
			return true, nil
		}
	}
	return false, nil
}

func (b *vLLMBackend) GetModelSize(model string) (int64, error) {
	models, err := b.ListModels()
	if err != nil {
		return 0, err
	}
	for _, m := range models {
		if m.Name == model {
			return m.Size, nil
		}
	}
	return 0, fmt.Errorf("model %s not found", model)
}