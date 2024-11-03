package openai_api

import (
	"encoding/json"
	"net/http"
)

// ErrorResponse represents an API error response
type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    int    `json:"code"`
	} `json:"error"`
}

// CreateErrorResponse creates a standardized error response
func CreateErrorResponse(message string, errType string) ErrorResponse {
	return ErrorResponse{
		Error: struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    int    `json:"code"`
		}{
			Message: message,
			Type:    errType,
			Code:    http.StatusBadRequest,
		},
	}
}

func V1Completions(w http.ResponseWriter, r *http.Request) {
	// Only allow POST method
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read the request body
	var requestData map[string]interface{}
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&requestData); err != nil {
		http.Error(w, "Error parsing JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Set response header
	w.Header().Set("Content-Type", "application/json")

	// Send back the same JSON data
	encoder := json.NewEncoder(w)
	if err := encoder.Encode(requestData); err != nil {
		http.Error(w, "Error encoding response: "+err.Error(), http.StatusInternalServerError)
		return
	}
}
