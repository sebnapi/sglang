package main

import (
	"fmt"
	"log"
	"net/http"
	"sglang/io"
	"sglang/openai_api"
	"time"
)

// Middleware for logging requests
func loggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		startTime := time.Now()
		log.Printf("Started %s %s", r.Method, r.URL.Path)
		next(w, r)
		log.Printf("Completed %s %s in %v", r.Method, r.URL.Path, time.Since(startTime))
	}
}

// Middleware for CORS
func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next(w, r)
	}
}

func main() {
	log.Println("Create IPC client and connect to Python server")
	client, err := io.NewClient()
	if err != nil {
		log.Fatalf("Failed to create IPC client: %v", err)
	}
	defer client.Close()

	if err := client.Connect(); err != nil {
		log.Fatalf("Failed to connect to IPC server: %v", err)
	}

	// Send test message
	msg := &io.IPCMessage{
		Type:    "SERVER_STARTED",
		Payload: []byte(fmt.Sprintf("GO-Server started on port %d", 40000)),
		Metadata: map[string]string{
			"timestamp": time.Now().String(),
		},
	}

	response, err := client.SendProto(msg)
	if err != nil {
		log.Printf("Failed to send IPC message: %v", err)
	} else {
		log.Printf("Received IPC response: type=%s metadata=%v", response.Type, response.Metadata)
	}

	// Define routes with middleware
	handler := openai_api.NewHandler(client)
	http.HandleFunc("/v1/completions", corsMiddleware(loggingMiddleware(handler.V1Completions)))

	// Set up the server
	port := 40000
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	// Start the server
	fmt.Printf("Server starting on port %d...\n", port)
	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
