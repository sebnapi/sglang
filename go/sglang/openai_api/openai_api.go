package openai_api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sglang/io"

	"google.golang.org/protobuf/proto"
)

type ErrorDetails struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    int    `json:"code"`
}

// ErrorResponse represents an API error response
type ErrorResponse struct {
	Error ErrorDetails `json:"error"`
}

func handleJSONError(err error) {
	if err == nil {
		return
	}
	switch e := err.(type) {
	case *json.SyntaxError:
		fmt.Printf("Syntax error at byte offset %d\n", e.Offset)
	case *json.UnmarshalTypeError:
		fmt.Printf("Invalid value for field %s\n", e.Field)
	default:
		fmt.Printf("Other error: %v\n", err)
	}
}

func CreateErrorResponse(message string, errType string) ErrorResponse {
	return ErrorResponse{
		Error: ErrorDetails{
			Message: message,
			Type:    errType,
			Code:    http.StatusBadRequest,
		},
	}
}

// Handler represents the HTTP handler with its dependencies
type Handler struct {
	ipcClient *io.Client
}

func NewHandler(ipcClient *io.Client) *Handler {
	return &Handler{
		ipcClient: ipcClient,
	}
}

// V1Completions handles the completions endpoint
func (h *Handler) V1Completions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read and parse the request body
	var rawRequest map[string]interface{}
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&rawRequest); err != nil {
		handleJSONError(err)
		errorResponse := CreateErrorResponse(
			"Failed to parse request data",
			"json_parsing_error",
		)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(errorResponse)
		return
	}
	defer r.Body.Close()

	// Convert the raw request into a CompletionRequest protobuf
	completionRequest := &io.CompletionRequest{}

	// Handle prompt which can be string or array
	if prompt, ok := rawRequest["prompt"].(string); ok {
		completionRequest.Prompt = &io.PromptContent{Text: &prompt}
	} else if promptArray, ok := rawRequest["prompt"].([]interface{}); ok {
		prompts := make([]string, len(promptArray))
		for i, p := range promptArray {
			if str, ok := p.(string); ok {
				prompts[i] = str
			}
		}
		completionRequest.Prompt = &io.PromptContent{Texts: prompts}
	}

	// Copy other fields
	if model, ok := rawRequest["model"].(string); ok {
		completionRequest.Model = model
	}
	if maxTokens, ok := rawRequest["max_tokens"].(float64); ok {
		completionRequest.MaxTokens = int32(maxTokens)
	}
	if temperature, ok := rawRequest["temperature"].(float64); ok {
		completionRequest.Temperature = float32(temperature)
	}
	if stream, ok := rawRequest["stream"].(bool); ok {
		completionRequest.Stream = stream
	}
	if topP, ok := rawRequest["top_p"].(float64); ok {
		completionRequest.TopP = float32(topP)
	}
	if regex, ok := rawRequest["regex"].(string); ok {
		completionRequest.Regex = &regex
	}

	// Create IPC message with CompletionRequest
	payloadBytes, err := proto.Marshal(completionRequest)
	if err != nil {
		errorResponse := CreateErrorResponse(
			"Failed to marshal request",
			"internal_error",
		)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(errorResponse)
		return
	}

	ipcMsg := &io.IPCMessage{
		Type:    "CompletionRequest",
		Payload: payloadBytes,
	}

	// Send IPC message and get response using the provided client
	response, err := h.ipcClient.SendProto(ipcMsg)
	if err != nil {
		errorResponse := CreateErrorResponse(
			"Failed to get completion response",
			"internal_error",
		)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(errorResponse)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(response.Payload)
}
