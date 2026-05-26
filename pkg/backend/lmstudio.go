package backend

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type LMStudioBackend struct {
	BaseURL string
	Client  *http.Client
}

func NewLMStudioBackend(baseURL string) *LMStudioBackend {
	return &LMStudioBackend{
		BaseURL: baseURL,
		Client: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

func (b *LMStudioBackend) Name() string { return "lmstudio" }

type LMModelInfo struct {
	ID         string `json:"id"`
	Object     string `json:"object"`
	CreatedAt  int64  `json:"created_at"`
	OwnedBy    string `json:"owned_by"`
	Permission []any  `json:"permission"`
}

type LMModelsResponse struct {
	Object string        `json:"object"`
	Data   []LMModelInfo `json:"data"`
}

func (b *LMStudioBackend) ListModels() ([]ModelInfo, error) {
	resp, err := b.Client.Get(b.BaseURL + "/v1/models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var modelsResp LMModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err != nil {
		return nil, err
	}

	var models []ModelInfo
	for _, m := range modelsResp.Data {
		models = append(models, ModelInfo{Name: m.ID})
	}
	return models, nil
}

type LMChatRequest struct {
	Model       string          `json:"model"`
	Messages    []LMChatMessage `json:"messages"`
	Stream      bool            `json:"stream"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
}

type LMChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type LMChatResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []LMChatChoice     `json:"choices"`
	Usage   LMTokensUsage      `json:"usage"`
}

type LMChatChoice struct {
	Index        int         `json:"index"`
	Message      LMMessage   `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type LMMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type LMTokensUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (b *LMStudioBackend) Chat(req ChatRequest) (*ChatResponse, error) {
	// Convert to LM Studio format
	lmReq := LMChatRequest{
		Model:   req.Model,
		Stream:  false,
		Messages: make([]LMChatMessage, len(req.Messages)),
	}
	for i, m := range req.Messages {
		lmReq.Messages[i] = LMChatMessage{Role: m.Role, Content: m.Content}
	}

	body, _ := json.Marshal(lmReq)
	httpReq, _ := http.NewRequest("POST", b.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.Client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var lmResp LMChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&lmResp); err != nil {
		return nil, err
	}

	if len(lmResp.Choices) == 0 {
		return nil, fmt.Errorf("no response from LM Studio")
	}

	return &ChatResponse{
		Model: lmResp.Model,
		Message: ChatMessage{
			Role:    lmResp.Choices[0].Message.Role,
			Content: lmResp.Choices[0].Message.Content,
		},
		Done: true,
	}, nil
}

func (b *LMStudioBackend) ChatStream(req ChatRequest) (io.ReadCloser, error) {
	lmReq := LMChatRequest{
		Model:   req.Model,
		Stream:  true,
		Messages: make([]LMChatMessage, len(req.Messages)),
	}
	for i, m := range req.Messages {
		lmReq.Messages[i] = LMChatMessage{Role: m.Role, Content: m.Content}
	}

	body, _ := json.Marshal(lmReq)
	httpReq, _ := http.NewRequest("POST", b.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.Client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	return resp.Body, nil
}

type LMLoadRequest struct {
	Model string `json:"model"`
}

func (b *LMStudioBackend) LoadModel(model string) error {
	body, _ := json.Marshal(LMLoadRequest{Model: model})
	httpReq, _ := http.NewRequest("POST", b.BaseURL+"/v1/models/load", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.Client.Do(httpReq)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func (b *LMStudioBackend) UnloadModel(model string) error {
	body, _ := json.Marshal(LMLoadRequest{Model: model})
	httpReq, _ := http.NewRequest("POST", b.BaseURL+"/v1/models/unload", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.Client.Do(httpReq)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

type LMPsResponse struct {
	Data []struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		BuiltIn bool   `json:"built_in"`
	} `json:"data"`
}

func (b *LMStudioBackend) IsRunning(model string) (bool, error) {
	resp, err := b.Client.Get(b.BaseURL + "/v1/models")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	var psResp LMPsResponse
	if err := json.NewDecoder(resp.Body).Decode(&psResp); err != nil {
		return false, err
	}

	for _, m := range psResp.Data {
		if m.ID == model {
			return true, nil
		}
	}
	return false, nil
}

func (b *LMStudioBackend) GetModelSize(model string) (int64, error) {
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
