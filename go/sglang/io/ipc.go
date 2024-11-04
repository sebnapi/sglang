package io

import (
	"fmt"

	zmq "github.com/pebbe/zmq4"
	"google.golang.org/protobuf/proto"
)

const (
	DefaultSocketPath = "ipc:///tmp/gopy-sglang-ipc.sock"
)

type Client struct {
	socket *zmq.Socket
	path   string
}

func NewClient() (*Client, error) {
	socketPath := DefaultSocketPath

	socket, err := zmq.NewSocket(zmq.REQ) // Go is requesting
	if err != nil {
		return nil, fmt.Errorf("failed to create socket: %w", err)
	}

	return &Client{
		socket: socket,
		path:   socketPath,
	}, nil
}

func (c *Client) Connect() error {
	if err := c.socket.Connect(c.path); err != nil {
		return fmt.Errorf("failed to connect to %s: %w", c.path, err)
	}
	return nil
}

func (c *Client) SendProto(msg *IPCMessage) (*IPCMessage, error) {
	// Serialize the protobuf message
	data, err := proto.Marshal(msg)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal protobuf: %w", err)
	}

	// Send the message
	if _, err := c.socket.SendBytes(data, 0); err != nil {
		return nil, fmt.Errorf("failed to send message: %w", err)
	}

	// Receive response
	responseData, err := c.socket.RecvBytes(0)
	if err != nil {
		return nil, fmt.Errorf("failed to receive response: %w", err)
	}

	// Unmarshal response
	response := &IPCMessage{}
	if err := proto.Unmarshal(responseData, response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return response, nil
}

func (c *Client) Close() error {
	return c.socket.Close()
}
