import zmq
import logging
import asyncio
from typing import Optional, Callable

from sglang.srt.openai_api.my_adapter_test import v1_completions
from sglang.srt.openai_api.protocol_pb2 import IPCMessage, CompletionRequest

from sglang.srt.openai_api.protocol import CompletionRequest as PyCompletionRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = "ipc:///tmp/gopy-sglang-ipc.sock"

class PyServer:
    def __init__(self, socket_path: Optional[str] = None,
                 message_handler: Callable = None):
        self.socket_path = socket_path or DEFAULT_SOCKET_PATH
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP) # Py is responding
        self.message_handler = message_handler
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def start(self):
        """Start the server and listen for protobuf messages from Go client."""
        try:
            self.socket.bind(self.socket_path)
            logger.info(f"Python server listening on {self.socket_path}")

            while True:
                # Wait for next request from Go client
                message_data = self.socket.recv()

                # Parse the protobuf message as IPC wrapper first
                ipc_message = IPCMessage()
                ipc_message.ParseFromString(message_data)
                logger.info(f"Received from Go: type={ipc_message.type}, metadata={ipc_message.metadata}")

                # Handle the message asynchronously
                response = self.loop.run_until_complete(self.message_handler(ipc_message))

                if response is not None:
                    self.socket.send(response.SerializeToString())
                    logger.info(f"Sent response to Go: type={response.type}")

        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        except Exception as e:
            logger.error(f"Error in server: {e}")
            raise
        finally:
            self.close()

    def close(self):
        """Close the socket and terminate context."""
        self.socket.close()
        self.context.term()
        self.loop.close()

class IPCMessageHandler:
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    async def __call__(self, ipc_message: IPCMessage):
        if ipc_message.type == "CompletionRequest":
            try:
                message = CompletionRequest()
                message.ParseFromString(ipc_message.payload)

                response = IPCMessage()
                response.type = "JsonResponse"

                py_message = PyCompletionRequest.from_proto(message)
                completion_response = await v1_completions(self.tokenizer_manager, py_message)
                response.payload = completion_response.model_dump_json().encode("utf-8")

                print(f"Parsed completion request: {message}")
                return response
            except Exception as e:
                logger.error(f"Failed to parse completion request: {e}")
                raise

        return ipc_message

if __name__ == "__main__":
    handler = IPCMessageHandler(tokenizer_manager=None)
    server = PyServer(message_handler=handler)
    print("Starting GO-PY IPC server... Press Ctrl+C to exit")
    server.start()
