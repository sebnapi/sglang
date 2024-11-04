package openai_api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Basic User struct for JSON examples
type User struct {
	Name    string   `json:"name"`
	Email   *string  `json:"email,omitempty"`
	Age     *int     `json:"age,omitempty"`
	Address Address  `json:"address,omitempty"`
	Tags    []string `json:"tags,omitempty"`
}

// Address struct for nested JSON example
type Address struct {
	Street string `json:"street"`
	City   string `json:"city"`
}

// Event struct with custom date handling
type Event struct {
	Name string    `json:"name"`
	Date time.Time `json:"date"`
}

// Custom unmarshaler for specific date format
func (e *Event) UnmarshalJSON(data []byte) error {
	type Alias Event
	aux := &struct {
		Date string `json:"date"`
		*Alias
	}{
		Alias: (*Alias)(e),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	var err error
	e.Date, err = time.Parse("2006-01-02", aux.Date)
	return err
}

// Example function showing different JSON parsing scenarios
func ExampleJSONParsing() {
	// 1. Basic JSON parsing
	jsonStr := `{"name": "Alice", "email": "alice@example.com", "age": 25}`
	var user User
	_ = json.Unmarshal([]byte(jsonStr), &user)

	// 2. Parsing JSON from request body
	handleRequest := func(r *http.Request) {
		var user User
		_ = json.NewDecoder(r.Body).Decode(&user)
	}
	_ = handleRequest // Prevent unused function warning

	// 3. Parsing into map[string]interface{}
	jsonMap := `{"name": "Alice", "details": {"city": "London", "active": true}}`
	var data map[string]interface{}
	_ = json.Unmarshal([]byte(jsonMap), &data)

	// 4. Array of structs
	jsonArray := `[
		{"name": "Alice", "age": 25},
		{"name": "Bob", "age": 30}
	]`
	var users []User
	_ = json.Unmarshal([]byte(jsonArray), &users)

	// 5. Nested JSON with arrays
	jsonNested := `{
		"name": "Alice",
		"address": {
			"street": "123 Main St",
			"city": "London"
		},
		"tags": ["admin", "active"]
	}`
	_ = json.Unmarshal([]byte(jsonNested), &user)
}

// Example function showing JSON parsing with error handling
func ExampleJSONParsingWithErrors() {
	// Helper function to handle JSON unmarshal errors
	handleJSONError := func(err error) {
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

	// 1. Basic error handling
	jsonStr := `{"name": "Alice", "email": "invalid-json`
	var user User
	handleJSONError(json.Unmarshal([]byte(jsonStr), &user))

	// 2. Type mismatch error handling
	jsonInvalid := `{"name": "Alice", "age": "not-a-number"}`
	handleJSONError(json.Unmarshal([]byte(jsonInvalid), &user))

	// 3. Required field validation
	jsonMissing := `{"email": "alice@example.com"}`
	handleJSONError(json.Unmarshal([]byte(jsonMissing), &user))

	// 4. Custom error response
	if err := json.Unmarshal([]byte(jsonInvalid), &user); err != nil {
		handleJSONError(err)
		errorResponse := CreateErrorResponse(
			"Failed to parse user data",
			"json_parsing_error",
		)
		jsonResponse, _ := json.Marshal(errorResponse)
		fmt.Printf("API Error Response: %s\n", string(jsonResponse))
	}
}
