"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
from typing import Dict, List, Optional, Union

from betterproto import Message, field

class ModelCard(Message):
    id: str = field(1)
    object: str = field(2, default="model")
    created: int = field(3, default=0)
    owned_by: str = field(4, default="sglang")
    root: Optional[str] = field(5)

    def __post_init__(self):
        if not self.created:
            self.created = int(time.time())

class ModelList(Message):
    object: str = field(1, default="list")
    data: List[ModelCard] = field(2, default_factory=list)

class ErrorResponse(Message):
    object: str = field(1, default="error")
    message: str = field(2)
    type: str = field(3)
    param: Optional[str] = field(4)
    code: int = field(5)

class LogProbs(Message):
    text_offset: List[int] = field(1, default_factory=list)
    token_logprobs: List[float] = field(2, default_factory=list)
    tokens: List[str] = field(3, default_factory=list)
    top_logprobs: List[Dict[str, float]] = field(4, default_factory=list)

class TopLogprob(Message):
    token: str = field(1)
    bytes: List[int] = field(2, default_factory=list)
    logprob: float = field(3)

class ChatCompletionTokenLogprob(Message):
    token: str = field(1)
    bytes: List[int] = field(2, default_factory=list)
    logprob: float = field(3)
    top_logprobs: List[TopLogprob] = field(4, default_factory=list)

class ChoiceLogprobs(Message):
    content: List[ChatCompletionTokenLogprob] = field(1, default_factory=list)

class UsageInfo(Message):
    prompt_tokens: int = field(1, default=0)
    total_tokens: int = field(2, default=0)
    completion_tokens: Optional[int] = field(3)
    prompt_tokens_details: Optional[Dict[str, int]] = field(4)

class StreamOptions(Message):
    include_usage: Optional[bool] = field(1)

class JsonSchema(Message):
    type: str = field(1)
    properties: Dict[str, str] = field(2)
    required: List[str] = field(3, default_factory=list)
    additional_properties: Optional[bool] = field(4)
    description: Optional[str] = field(5)

class JsonSchemaResponseFormat(Message):
    name: str = field(1)
    description: Optional[str] = field(2)
    schema: Optional[JsonSchema] = field(3)
    strict: Optional[bool] = field(4)

class PromptContent(Message):
    text: Optional[str] = field(1)  # Single string
    texts: List[str] = field(2, default_factory=list)  # Multiple strings
    tokens: List[int] = field(3, default_factory=list)  # Single sequence of tokens
    token_matrix: List[List[int]] = field(4, default_factory=list)  # Multiple sequences of tokens

class MatchedStop(Message):
    int_value: Optional[int] = field(1)
    str_value: Optional[str] = field(2)

class StopTrimConfig(Message):
    single: bool = field(1, default=False)
    multiple: List[bool] = field(2, default_factory=list)

class CompletionRequest(Message):
    model: str = field(1)
    prompt: PromptContent = field(2)
    best_of: Optional[int] = field(3)
    echo: bool = field(4, default=False)
    frequency_penalty: float = field(5, default=0.0)
    logit_bias: Optional[Dict[str, float]] = field(6)
    logprobs: Optional[int] = field(7)
    max_tokens: int = field(8, default=16)
    n: int = field(9, default=1)
    presence_penalty: float = field(10, default=0.0)
    seed: Optional[int] = field(11)
    stop: List[str] = field(12, default_factory=list)
    stream: bool = field(13, default=False)
    stream_options: Optional[StreamOptions] = field(14)
    suffix: Optional[str] = field(15)
    temperature: float = field(16, default=1.0)
    top_p: float = field(17, default=1.0)
    user: Optional[str] = field(18)
    regex: Optional[str] = field(19)
    json_schema: Optional[str] = field(20)
    ignore_eos: bool = field(21, default=False)
    min_tokens: int = field(22, default=0)
    repetition_penalty: float = field(23, default=1.0)
    stop_token_ids: List[int] = field(24, default_factory=list)
    no_stop_trim: StopTrimConfig = field(25)

class CompletionResponseChoice(Message):
    index: int = field(1, default=0)
    text: str = field(2)
    logprobs: Optional[LogProbs] = field(3)
    finish_reason: Optional[str] = field(4)
    matched_stop: Optional[MatchedStop] = field(5)

class CompletionResponse(Message):
    id: str = field(1)
    object: str = field(2, default="text_completion")
    created: int = field(3, default=0)
    model: str = field(4)
    choices: List[CompletionResponseChoice] = field(5, default_factory=list)
    usage: UsageInfo = field(6)

    def __post_init__(self):
        if not self.created:
            self.created = int(time.time())

class ChatMessage(Message):
    role: Optional[str] = field(1)
    content: Optional[str] = field(2)

class ChatCompletionResponseChoice(Message):
    index: int = field(1, default=0)
    message: ChatMessage = field(2)
    logprobs: Optional[LogProbsUnion] = field(3)
    finish_reason: str = field(4)
    matched_stop: Optional[MatchedStop] = field(5)

class ChatCompletionResponse(Message):
    id: str = field(1)
    object: str = field(2, default="chat.completion")
    created: int = field(3, default=0)
    model: str = field(4)
    choices: List[ChatCompletionResponseChoice] = field(5, default_factory=list)
    usage: UsageInfo = field(6)

    def __post_init__(self):
        if not self.created:
            self.created = int(time.time())

class FileRequest(Message):
    file: bytes = field(1)
    purpose: str = field(2, default="batch")

class FileResponse(Message):
    id: str = field(1)
    object: str = field(2, default="file")
    bytes: int = field(3)
    created_at: int = field(4)
    filename: str = field(5)
    purpose: str = field(6)

class FileDeleteResponse(Message):
    id: str = field(1)
    object: str = field(2, default="file")
    deleted: bool = field(3)

class BatchMetadata(Message):
    key: str = field(1)
    value: str = field(2)

class BatchError(Message):
    code: str = field(1)
    message: str = field(2)

class BatchRequest(Message):
    input_file_id: str = field(1)
    endpoint: str = field(2)
    completion_window: str = field(3)
    metadata: List[BatchMetadata] = field(4, default_factory=list)

class RequestCounts(Message):
    total: int = field(1, default=0)
    completed: int = field(2, default=0)
    failed: int = field(3, default=0)

class BatchResponse(Message):
    id: str = field(1)
    object: str = field(2, default="batch")
    endpoint: str = field(3)
    errors: Optional[BatchError] = field(4)
    input_file_id: str = field(5)
    completion_window: str = field(6)
    status: str = field(7, default="validating")
    output_file_id: Optional[str] = field(8)
    error_file_id: Optional[str] = field(9)
    created_at: int = field(10)
    in_progress_at: Optional[int] = field(11)
    expires_at: Optional[int] = field(12)
    finalizing_at: Optional[int] = field(13)
    completed_at: Optional[int] = field(14)
    failed_at: Optional[int] = field(15)
    expired_at: Optional[int] = field(16)
    cancelling_at: Optional[int] = field(17)
    cancelled_at: Optional[int] = field(18)
    request_counts: RequestCounts = field(19)
    metadata: List[BatchMetadata] = field(20, default_factory=list)

class CompletionResponseStreamChoice(Message):
    index: int = field(1, default=0)
    text: str = field(2)
    logprobs: Optional[LogProbs] = field(3)
    finish_reason: Optional[str] = field(4)
    matched_stop: Optional[MatchedStop] = field(5)

class CompletionStreamResponse(Message):
    id: str = field(1)
    object: str = field(2, default="text_completion")
    created: int = field(3, default=0)
    model: str = field(4)
    choices: List[CompletionResponseStreamChoice] = field(5, default_factory=list)
    usage: Optional[UsageInfo] = field(6)

    def __post_init__(self):
        if not self.created:
            self.created = int(time.time())

class ChatCompletionMessageContentTextPart(Message):
    type: str = field(1, default="text")
    text: str = field(2)

class ChatCompletionMessageContentImageURL(Message):
    url: str = field(1)
    detail: str = field(2, default="auto")

class ChatCompletionMessageContentImagePart(Message):
    type: str = field(1, default="image_url")
    image_url: ChatCompletionMessageContentImageURL = field(2)
    modalities: str = field(3, default="image")

class GenericMessageContent(Message):
    text: Optional[str] = field(1)  # Simple string content
    parts: List[ChatCompletionMessageContentTextPart] = field(2, default_factory=list)  # Text parts

class ChatCompletionMessageGenericParam(Message):
    role: str = field(1)
    content: GenericMessageContent = field(2)

class MessageContent(Message):
    text: Optional[str] = field(1)  # Simple string content
    parts: List[ChatCompletionMessageContentTextPart] = field(2, default_factory=list)  # Text parts
    images: List[ChatCompletionMessageContentImagePart] = field(3, default_factory=list)  # Image parts

class ChatCompletionMessageUserParam(Message):
    role: str = field(1, default="user")
    content: MessageContent = field(2)

class ResponseFormat(Message):
    type: str = field(1, default="text")
    json_schema: Optional[JsonSchemaResponseFormat] = field(2)

class MessageParam(Message):
    generic_param: Optional[ChatCompletionMessageGenericParam] = field(1)
    user_param: Optional[ChatCompletionMessageUserParam] = field(2)

class ChatCompletionRequest(Message):
    messages: List[MessageParam] = field(1, default_factory=list)
    model: str = field(2)
    frequency_penalty: float = field(3, default=0.0)
    logit_bias: Optional[Dict[str, float]] = field(4)
    logprobs: bool = field(5, default=False)
    top_logprobs: Optional[int] = field(6)
    max_tokens: Optional[int] = field(7)
    n: int = field(8, default=1)
    presence_penalty: float = field(9, default=0.0)
    response_format: Optional[ResponseFormat] = field(10)
    seed: Optional[int] = field(11)
    stop: List[str] = field(12, default_factory=list)
    stream: bool = field(13, default=False)
    stream_options: Optional[StreamOptions] = field(14)
    temperature: float = field(15, default=0.7)
    top_p: float = field(16, default=1.0)
    user: Optional[str] = field(17)
    regex: Optional[str] = field(18)
    min_tokens: int = field(19, default=0)
    repetition_penalty: float = field(20, default=1.0)
    stop_token_ids: List[int] = field(21, default_factory=list)
    ignore_eos: bool = field(22, default=False)

class DeltaMessage(Message):
    role: Optional[str] = field(1)
    content: Optional[str] = field(2)

class LogProbsUnion(Message):
    basic: Optional[LogProbs] = field(1)
    choice: Optional[ChoiceLogprobs] = field(2)

class ChatCompletionResponseStreamChoice(Message):
    index: int = field(1, default=0)
    delta: DeltaMessage = field(2)
    logprobs: Optional[LogProbsUnion] = field(3)
    finish_reason: Optional[str] = field(4)
    matched_stop: Optional[MatchedStop] = field(5)

class ChatCompletionStreamResponse(Message):
    id: str = field(1)
    object: str = field(2, default="chat.completion.chunk")
    created: int = field(3, default=0)
    model: str = field(4)
    choices: List[ChatCompletionResponseStreamChoice] = field(5, default_factory=list)
    usage: Optional[UsageInfo] = field(6)

    def __post_init__(self):
        if not self.created:
            self.created = int(time.time())

class EmbeddingInput(Message):
    text: Optional[str] = field(1)  # Single string
    texts: List[str] = field(2, default_factory=list)  # Multiple strings
    tokens: List[int] = field(3, default_factory=list)  # Single sequence of tokens
    token_matrix: List[List[int]] = field(4, default_factory=list)  # Multiple sequences of tokens

class EmbeddingRequest(Message):
    input: EmbeddingInput = field(1)
    model: str = field(2)
    encoding_format: str = field(3, default="float")
    dimensions: Optional[int] = field(4)
    user: Optional[str] = field(5)

class EmbeddingObject(Message):
    embedding: List[float] = field(1, default_factory=list)
    index: int = field(2, default=0)
    object: str = field(3, default="embedding")

class EmbeddingResponse(Message):
    data: List[EmbeddingObject] = field(1, default_factory=list)
    model: str = field(2)
    object: str = field(3, default="list")
    usage: Optional[UsageInfo] = field(4)
