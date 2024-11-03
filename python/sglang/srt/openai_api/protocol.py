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

"""Pydantic models for OpenAI API protocol"""

import abc
import time
import json
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal
from sglang.srt.openai_api import protocol_pb2


class ProtoConvertible(abc.ABC):
    @abc.abstractmethod
    def to_proto(self):
        pass

    @classmethod
    @abc.abstractmethod
    def from_proto(cls, proto) -> "ProtoConvertible":
        pass


class ModelCard(BaseModel, ProtoConvertible):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: Optional[str] = None

    def to_proto(self):
        return protocol_pb2.ModelCard(
            id=self.id,
            object=self.object,
            created=self.created,
            owned_by=self.owned_by,
            root=self.root
        )

    @classmethod
    def from_proto(cls, proto) -> "ModelCard":
        return cls(
            id=proto.id,
            object=proto.object,
            created=proto.created,
            owned_by=proto.owned_by,
            root=proto.root
        )


class ModelList(BaseModel, ProtoConvertible):
    """Model list consists of model cards."""

    object: str = "list"
    data: List[ModelCard] = []

    def to_proto(self):
        return protocol_pb2.ModelList(
            object=self.object,
            data=[card.to_proto() for card in self.data]
        )

    @classmethod
    def from_proto(cls, proto) -> "ModelList":
        return cls(
            object=proto.object,
            data=[ModelCard.from_proto(card) for card in proto.data]
        )


class ErrorResponse(BaseModel, ProtoConvertible):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int

    def to_proto(self):
        return protocol_pb2.ErrorResponse(
            object=self.object,
            message=self.message,
            type=self.type,
            param=self.param,
            code=self.code
        )

    @classmethod
    def from_proto(cls, proto) -> "ErrorResponse":
        return cls(
            object=proto.object,
            message=proto.message,
            type=proto.type,
            param=proto.param,
            code=proto.code
        )


class LogProbs(BaseModel, ProtoConvertible):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)

    def to_proto(self):
        # Convert list of dicts to list of TokenLogprobs
        top_logprobs_proto = []
        for logprob_dict in self.top_logprobs:
            if logprob_dict is not None:
                token_logprobs = protocol_pb2.TokenLogprobs(values=logprob_dict)
                top_logprobs_proto.append(token_logprobs)
            else:
                top_logprobs_proto.append(protocol_pb2.TokenLogprobs())

        return protocol_pb2.LogProbs(
            text_offset=self.text_offset,
            token_logprobs=[lp for lp in self.token_logprobs if lp is not None],
            tokens=self.tokens,
            top_logprobs=top_logprobs_proto
        )

    @classmethod
    def from_proto(cls, proto) -> "LogProbs":
        # Convert list of TokenLogprobs back to list of dicts
        top_logprobs = [
            logprob.values if logprob.values else None
            for logprob in proto.top_logprobs
        ]

        return cls(
            text_offset=proto.text_offset,
            token_logprobs=proto.token_logprobs,
            tokens=proto.tokens,
            top_logprobs=top_logprobs
        )


class TopLogprob(BaseModel, ProtoConvertible):
    token: str
    bytes: List[int]
    logprob: float

    def to_proto(self):
        return protocol_pb2.TopLogprob(
            token=self.token,
            bytes=self.bytes,
            logprob=self.logprob
        )

    @classmethod
    def from_proto(cls, proto) -> "TopLogprob":
        return cls(
            token=proto.token,
            bytes=proto.bytes,
            logprob=proto.logprob
        )


class ChatCompletionTokenLogprob(BaseModel, ProtoConvertible):
    token: str
    bytes: List[int]
    logprob: float
    top_logprobs: List[TopLogprob]

    def to_proto(self):
        return protocol_pb2.ChatCompletionTokenLogprob(
            token=self.token,
            bytes=self.bytes,
            logprob=self.logprob,
            top_logprobs=[lp.to_proto() for lp in self.top_logprobs]
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionTokenLogprob":
        return cls(
            token=proto.token,
            bytes=proto.bytes,
            logprob=proto.logprob,
            top_logprobs=[TopLogprob.from_proto(lp) for lp in proto.top_logprobs]
        )


class ChoiceLogprobs(BaseModel, ProtoConvertible):
    # build for v1/chat/completions response
    content: List[ChatCompletionTokenLogprob]

    def to_proto(self):
        return protocol_pb2.ChoiceLogprobs(
            content=[lp.to_proto() for lp in self.content]
        )

    @classmethod
    def from_proto(cls, proto) -> "ChoiceLogprobs":
        return cls(
            content=[ChatCompletionTokenLogprob.from_proto(lp) for lp in proto.content]
        )


class UsageInfo(BaseModel, ProtoConvertible):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    # only used to return cached tokens when --enable-cache-report is set
    prompt_tokens_details: Optional[Dict[str, int]] = None

    def to_proto(self):
        return protocol_pb2.UsageInfo(
            prompt_tokens=self.prompt_tokens,
            total_tokens=self.total_tokens,
            completion_tokens=self.completion_tokens,
            prompt_tokens_details=self.prompt_tokens_details
        )

    @classmethod
    def from_proto(cls, proto) -> "UsageInfo":
        return cls(
            prompt_tokens=proto.prompt_tokens,
            total_tokens=proto.total_tokens,
            completion_tokens=proto.completion_tokens,
            prompt_tokens_details=proto.prompt_tokens_details
        )


class StreamOptions(BaseModel, ProtoConvertible):
    include_usage: Optional[bool] = False

    def to_proto(self):
        return protocol_pb2.StreamOptions(
            include_usage=self.include_usage
        )

    @classmethod
    def from_proto(cls, proto) -> "StreamOptions":
        return cls(
            include_usage=proto.include_usage
        )


class JsonSchemaResponseFormat(BaseModel, ProtoConvertible):
    name: str
    description: Optional[str] = None
    # use alias to workaround pydantic conflict
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
    strict: Optional[bool] = False

    def to_proto(self):
        return protocol_pb2.JsonSchemaResponseFormat(
            name=self.name,
            description=self.description,
            schema=json.dumps(self.schema_) if self.schema_ is not None else None,
            strict=self.strict
        )

    @classmethod
    def from_proto(cls, proto) -> "JsonSchemaResponseFormat":
        return cls(
            name=proto.name,
            description=proto.description,
            schema=json.loads(proto.schema) if proto.schema else None,
            strict=proto.strict
        )


class FileRequest(BaseModel, ProtoConvertible):
    # https://platform.openai.com/docs/api-reference/files/create
    file: bytes  # The File object (not file name) to be uploaded
    purpose: str = (
        "batch"  # The intended purpose of the uploaded file, default is "batch"
    )

    def to_proto(self):
        return protocol_pb2.FileRequest(
            file=self.file,
            purpose=self.purpose
        )

    @classmethod
    def from_proto(cls, proto) -> "FileRequest":
        return cls(
            file=proto.file,
            purpose=proto.purpose
        )


class FileResponse(BaseModel, ProtoConvertible):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str

    def to_proto(self):
        return protocol_pb2.FileResponse(
            id=self.id,
            object=self.object,
            bytes=self.bytes,
            created_at=self.created_at,
            filename=self.filename,
            purpose=self.purpose
        )

    @classmethod
    def from_proto(cls, proto) -> "FileResponse":
        return cls(
            id=proto.id,
            object=proto.object,
            bytes=proto.bytes,
            created_at=proto.created_at,
            filename=proto.filename,
            purpose=proto.purpose
        )


class FileDeleteResponse(BaseModel, ProtoConvertible):
    id: str
    object: str = "file"
    deleted: bool

    def to_proto(self):
        return protocol_pb2.FileDeleteResponse(
            id=self.id,
            object=self.object,
            deleted=self.deleted
        )

    @classmethod
    def from_proto(cls, proto) -> "FileDeleteResponse":
        return cls(
            id=proto.id,
            object=proto.object,
            deleted=proto.deleted
        )

class BatchMetadata(BaseModel, ProtoConvertible):
    key: str
    value: str

    def to_proto(self):
        return protocol_pb2.BatchMetadata(
            key=self.key,
            value=self.value
        )

    @classmethod
    def from_proto(cls, proto) -> "BatchMetadata":
        return cls(
            key=proto.key,
            value=proto.value
        )

class BatchRequest(BaseModel, ProtoConvertible):
    input_file_id: (
        str  # The ID of an uploaded file that contains requests for the new batch
    )
    endpoint: str  # The endpoint to be used for all requests in the batch
    completion_window: str  # The time frame within which the batch should be processed
    metadata: Optional[List[BatchMetadata]] = None  # Optional custom metadata for the batch

    def to_proto(self):
        return protocol_pb2.BatchRequest(
            input_file_id=self.input_file_id,
            endpoint=self.endpoint,
            completion_window=self.completion_window,
            metadata=[m.to_proto() for m in self.metadata] if self.metadata else []
        )

    @classmethod
    def from_proto(cls, proto) -> "BatchRequest":
        return cls(
            input_file_id=proto.input_file_id,
            endpoint=proto.endpoint,
            completion_window=proto.completion_window,
            metadata=[BatchMetadata.from_proto(m) for m in proto.metadata] if proto.metadata else None
        )

class BatchError(BaseModel, ProtoConvertible):
    code: str
    message: str

    def to_proto(self):
        return protocol_pb2.BatchError(
            code=self.code,
            message=self.message
        )

    @classmethod
    def from_proto(cls, proto) -> "BatchError":
        return cls(
            code=proto.code,
            message=proto.message
        )

class BatchResponse(BaseModel, ProtoConvertible):
    id: str
    object: str = "batch"
    endpoint: str
    errors: Optional[BatchError] = None
    input_file_id: str
    completion_window: str
    status: str = "validating"
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: dict = {"total": 0, "completed": 0, "failed": 0}
    metadata: Optional[List[BatchMetadata]] = None

    def to_proto(self):
        request_counts = protocol_pb2.RequestCounts(
            total=self.request_counts.get("total", 0),
            completed=self.request_counts.get("completed", 0),
            failed=self.request_counts.get("failed", 0)
        )

        return protocol_pb2.BatchResponse(
            id=self.id,
            object=self.object,
            endpoint=self.endpoint,
            errors=self.errors.to_proto() if self.errors else None,
            input_file_id=self.input_file_id,
            completion_window=self.completion_window,
            status=self.status,
            output_file_id=self.output_file_id,
            error_file_id=self.error_file_id,
            created_at=self.created_at,
            in_progress_at=self.in_progress_at,
            expires_at=self.expires_at,
            finalizing_at=self.finalizing_at,
            completed_at=self.completed_at,
            failed_at=self.failed_at,
            expired_at=self.expired_at,
            cancelling_at=self.cancelling_at,
            cancelled_at=self.cancelled_at,
            request_counts=request_counts,
            metadata=[m.to_proto() for m in self.metadata] if self.metadata else []
        )

    @classmethod
    def from_proto(cls, proto) -> "BatchResponse":
        request_counts = {
            "total": proto.request_counts.total,
            "completed": proto.request_counts.completed,
            "failed": proto.request_counts.failed
        }

        return cls(
            id=proto.id,
            object=proto.object,
            endpoint=proto.endpoint,
            errors=BatchError.from_proto(proto.errors) if proto.errors else None,
            input_file_id=proto.input_file_id,
            completion_window=proto.completion_window,
            status=proto.status,
            output_file_id=proto.output_file_id,
            error_file_id=proto.error_file_id,
            created_at=proto.created_at,
            in_progress_at=proto.in_progress_at,
            expires_at=proto.expires_at,
            finalizing_at=proto.finalizing_at,
            completed_at=proto.completed_at,
            failed_at=proto.failed_at,
            expired_at=proto.expired_at,
            cancelling_at=proto.cancelling_at,
            cancelled_at=proto.cancelled_at,
            request_counts=request_counts,
            metadata=[BatchMetadata.from_proto(m) for m in proto.metadata] if proto.metadata else None
        )

class StopTrimConfig(BaseModel, ProtoConvertible):
    single: bool = False
    multiple: List[bool] = Field(default_factory=list)

    def to_proto(self):
        return protocol_pb2.StopTrimConfig(
            single=self.single,
            multiple=self.multiple
        )

    @classmethod
    def from_proto(cls, proto) -> "StopTrimConfig":
        return cls(
            single=proto.single,
            multiple=proto.multiple
        )

class CompletionRequest(BaseModel, ProtoConvertible):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    regex: Optional[str] = None
    json_schema: Optional[str] = None
    ignore_eos: bool = False
    min_tokens: int = 0
    repetition_penalty: Optional[float] = 1.0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    no_stop_trim: Union[bool, List[bool]] = False

    def to_proto(self):
        prompt_content = protocol_pb2.PromptContent()
        if isinstance(self.prompt, str):
            prompt_content.text = self.prompt
        elif isinstance(self.prompt, list):
            if all(isinstance(x, int) for x in self.prompt):
                prompt_content.tokens.extend(self.prompt)
            elif all(isinstance(x, list) for x in self.prompt):
                # Convert each inner list to a TokenSequence message
                token_lists = [protocol_pb2.TokenSequence(tokens=tokens) for tokens in self.prompt]
                prompt_content.token_matrix.extend(token_lists)
            else:
                prompt_content.texts.extend(self.prompt)

        no_stop_trim = protocol_pb2.StopTrimConfig(
            single=self.no_stop_trim if isinstance(self.no_stop_trim, bool) else False,
            multiple=self.no_stop_trim if isinstance(self.no_stop_trim, list) else []
        )

        stop_list = []
        if isinstance(self.stop, str):
            stop_list.append(self.stop)
        elif isinstance(self.stop, list):
            stop_list.extend(self.stop)

        return protocol_pb2.CompletionRequest(
            model=self.model,
            prompt=prompt_content,
            best_of=self.best_of,
            echo=self.echo,
            frequency_penalty=self.frequency_penalty,
            logit_bias=self.logit_bias,
            logprobs=self.logprobs,
            max_tokens=self.max_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=stop_list,
            stream=self.stream,
            stream_options=self.stream_options.to_proto() if self.stream_options else None,
            suffix=self.suffix,
            temperature=self.temperature,
            top_p=self.top_p,
            user=self.user,
            regex=self.regex,
            json_schema=self.json_schema,
            ignore_eos=self.ignore_eos,
            min_tokens=self.min_tokens,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=self.stop_token_ids,
            no_stop_trim=no_stop_trim
        )

    @classmethod
    def from_proto(cls, proto) -> "CompletionRequest":
        # Convert prompt from proto format
        if proto.prompt.text:
            prompt = proto.prompt.text
        elif proto.prompt.texts:
            prompt = proto.prompt.texts
        elif proto.prompt.tokens:
            prompt = proto.prompt.tokens
        else:
            prompt = [list(token_list.tokens) for token_list in proto.prompt.token_matrix]

        # Convert no_stop_trim
        no_stop_trim = proto.no_stop_trim.multiple if proto.no_stop_trim.multiple else proto.no_stop_trim.single

        return cls(
            model=proto.model,
            prompt=prompt,
            best_of=proto.best_of,
            echo=proto.echo,
            frequency_penalty=proto.frequency_penalty,
            logit_bias=proto.logit_bias,
            logprobs=proto.logprobs,
            max_tokens=proto.max_tokens,
            n=proto.n,
            presence_penalty=proto.presence_penalty,
            seed=proto.seed,
            stop=proto.stop,
            stream=proto.stream,
            stream_options=StreamOptions.from_proto(proto.stream_options) if proto.stream_options else None,
            suffix=proto.suffix,
            temperature=proto.temperature,
            top_p=proto.top_p,
            user=proto.user,
            regex=proto.regex,
            json_schema=proto.json_schema,
            ignore_eos=proto.ignore_eos,
            min_tokens=proto.min_tokens,
            repetition_penalty=proto.repetition_penalty,
            stop_token_ids=proto.stop_token_ids,
            no_stop_trim=no_stop_trim
        )


class CompletionResponseChoice(BaseModel, ProtoConvertible):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None

    def to_proto(self):
        matched_stop = None
        if isinstance(self.matched_stop, int):
            matched_stop = protocol_pb2.MatchedStop(int_value=self.matched_stop)
        elif isinstance(self.matched_stop, str):
            matched_stop = protocol_pb2.MatchedStop(str_value=self.matched_stop)

        return protocol_pb2.CompletionResponseChoice(
            index=self.index,
            text=self.text,
            logprobs=self.logprobs.to_proto() if self.logprobs else None,
            finish_reason=self.finish_reason,
            matched_stop=matched_stop
        )

    @classmethod
    def from_proto(cls, proto) -> "CompletionResponseChoice":
        matched_stop = None
        if proto.matched_stop:
            if proto.matched_stop.HasField('int_value'):
                matched_stop = proto.matched_stop.int_value
            elif proto.matched_stop.HasField('str_value'):
                matched_stop = proto.matched_stop.str_value

        return cls(
            index=proto.index,
            text=proto.text,
            logprobs=LogProbs.from_proto(proto.logprobs) if proto.logprobs else None,
            finish_reason=proto.finish_reason,
            matched_stop=matched_stop
        )


class CompletionResponse(BaseModel, ProtoConvertible):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo

    def to_proto(self):
        return protocol_pb2.CompletionResponse(
            id=self.id,
            object=self.object,
            created=self.created,
            model=self.model,
            choices=[choice.to_proto() for choice in self.choices],
            usage=self.usage.to_proto()
        )

    @classmethod
    def from_proto(cls, proto) -> "CompletionResponse":
        return cls(
            id=proto.id,
            object=proto.object,
            created=proto.created,
            model=proto.model,
            choices=[CompletionResponseChoice.from_proto(choice) for choice in proto.choices],
            usage=UsageInfo.from_proto(proto.usage)
        )


class CompletionResponseStreamChoice(BaseModel, ProtoConvertible):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None

    def to_proto(self):
        logprobs_proto = None
        if self.logprobs:
            logprobs_proto = self.logprobs.to_proto()

        matched_stop = None
        if isinstance(self.matched_stop, int):
            matched_stop = protocol_pb2.MatchedStop(int_value=self.matched_stop)
        elif isinstance(self.matched_stop, str):
            matched_stop = protocol_pb2.MatchedStop(str_value=self.matched_stop)

        return protocol_pb2.CompletionResponseStreamChoice(
            index=self.index,
            text=self.text,
            logprobs=logprobs_proto,
            finish_reason=self.finish_reason,
            matched_stop=matched_stop
        )

    @classmethod
    def from_proto(cls, proto) -> "CompletionResponseStreamChoice":
        logprobs = None
        if proto.logprobs:
            logprobs = LogProbs.from_proto(proto.logprobs)

        matched_stop = None
        if proto.matched_stop:
            if proto.matched_stop.int_value is not None:
                matched_stop = proto.matched_stop.int_value
            elif proto.matched_stop.str_value is not None:
                matched_stop = proto.matched_stop.str_value

        return cls(
            index=proto.index,
            text=proto.text,
            logprobs=logprobs,
            finish_reason=proto.finish_reason,
            matched_stop=matched_stop
        )


class CompletionStreamResponse(BaseModel, ProtoConvertible):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None

    def to_proto(self):
        return protocol_pb2.CompletionStreamResponse(
            id=self.id,
            object=self.object,
            created=self.created,
            model=self.model,
            choices=[choice.to_proto() for choice in self.choices],
            usage=self.usage.to_proto() if self.usage else None
        )

    @classmethod
    def from_proto(cls, proto) -> "CompletionStreamResponse":
        return cls(
            id=proto.id,
            object=proto.object,
            created=proto.created,
            model=proto.model,
            choices=[CompletionResponseStreamChoice.from_proto(choice) for choice in proto.choices],
            usage=UsageInfo.from_proto(proto.usage) if proto.usage else None
        )


class ChatCompletionMessageContentTextPart(BaseModel, ProtoConvertible):
    type: Literal["text"]
    text: str

    def to_proto(self):
        return protocol_pb2.ChatCompletionMessageContentTextPart(
            type=self.type,
            text=self.text
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionMessageContentTextPart":
        return cls(
            type=proto.type,
            text=proto.text
        )

class ChatCompletionMessageContentImageURL(BaseModel, ProtoConvertible):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"

    def to_proto(self):
        return protocol_pb2.ChatCompletionMessageContentImageURL(
            url=self.url,
            detail=self.detail
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionMessageContentImageURL":
        return cls(
            url=proto.url,
            detail=proto.detail
        )


class ChatCompletionMessageContentImagePart(BaseModel, ProtoConvertible):
    type: Literal["image_url"]
    image_url: ChatCompletionMessageContentImageURL
    modalities: Optional[Literal["image", "multi-images", "video"]] = "image"

    def to_proto(self):
        return protocol_pb2.ChatCompletionMessageContentImagePart(
            type=self.type,
            image_url=self.image_url.to_proto(),
            modalities=self.modalities
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionMessageContentImagePart":
        return cls(
            type=proto.type,
            image_url=ChatCompletionMessageContentImageURL.from_proto(proto.image_url),
            modalities=proto.modalities
        )


ChatCompletionMessageContentPart = Union[
    ChatCompletionMessageContentTextPart, ChatCompletionMessageContentImagePart
]


class ChatCompletionMessageGenericParam(BaseModel, ProtoConvertible):
    role: Literal["system", "assistant"]
    content: Union[str, List[ChatCompletionMessageContentTextPart]]

    def to_proto(self):
        content = protocol_pb2.GenericMessageContent()
        if isinstance(self.content, str):
            content.text = self.content
        else:
            content.parts.extend([p.to_proto() for p in self.content])

        return protocol_pb2.ChatCompletionMessageGenericParam(
            role=self.role,
            content=content
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionMessageGenericParam":
        if proto.content.text:
            content = proto.content.text
        else:
            content = [ChatCompletionMessageContentTextPart.from_proto(p) for p in proto.content.parts]

        return cls(
            role=proto.role,
            content=content
        )


class ChatCompletionMessageUserParam(BaseModel, ProtoConvertible):
    role: Literal["user"]
    content: Union[str, List[ChatCompletionMessageContentPart]]

    def to_proto(self):
        content = protocol_pb2.MessageContent()
        if isinstance(self.content, str):
            content.text = self.content
        else:
            for p in self.content:
                if isinstance(p, ChatCompletionMessageContentTextPart):
                    content.parts.append(p.to_proto())
                elif isinstance(p, ChatCompletionMessageContentImagePart):
                    content.images.append(p.to_proto())

        return protocol_pb2.ChatCompletionMessageUserParam(
            role=self.role,
            content=content
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionMessageUserParam":
        if proto.content.text:
            content = proto.content.text
        else:
            content = []
            content.extend([ChatCompletionMessageContentTextPart.from_proto(p)
                             for p in proto.content.parts])
            content.extend([ChatCompletionMessageContentImagePart.from_proto(p)
                             for p in proto.content.images])

        return cls(
            role=proto.role,
            content=content
        )


ChatCompletionMessageParam = Union[
    ChatCompletionMessageGenericParam, ChatCompletionMessageUserParam
]


class ResponseFormat(BaseModel, ProtoConvertible):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None

    def to_proto(self):
        return protocol_pb2.ResponseFormat(
            type=self.type,
            json_schema=self.json_schema.to_proto() if self.json_schema else None
        )

    @classmethod
    def from_proto(cls, proto) -> "ResponseFormat":
        return cls(
            type=proto.type,
            json_schema=JsonSchemaResponseFormat.from_proto(proto.json_schema) if proto.json_schema else None
        )


class ChatCompletionRequest(BaseModel, ProtoConvertible):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    regex: Optional[str] = None
    min_tokens: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    ignore_eos: bool = False

    def to_proto(self):
        stop_list = []
        if isinstance(self.stop, str):
            stop_list.append(self.stop)
        elif isinstance(self.stop, list):
            stop_list.extend(self.stop)

        messages_proto = []
        for msg in self.messages:
            if isinstance(msg, ChatCompletionMessageGenericParam):
                content = protocol_pb2.GenericMessageContent()
                if isinstance(msg.content, str):
                    content.text = msg.content
                else:
                    content.parts.extend([p.to_proto() for p in msg.content])
                messages_proto.append(protocol_pb2.MessageParam(
                    generic_param=protocol_pb2.ChatCompletionMessageGenericParam(
                        role=msg.role,
                        content=content
                    )
                ))
            else:  # UserParam
                content = protocol_pb2.MessageContent()
                if isinstance(msg.content, str):
                    content.text = msg.content
                else:
                    for p in msg.content:
                        if isinstance(p, ChatCompletionMessageContentTextPart):
                            content.parts.append(p.to_proto())
                        elif isinstance(p, ChatCompletionMessageContentImagePart):
                                content.images.append(p.to_proto())
                messages_proto.append(protocol_pb2.MessageParam(
                    user_param=protocol_pb2.ChatCompletionMessageUserParam(
                        role=msg.role,
                        content=content
                    )
                ))

        return protocol_pb2.ChatCompletionRequest(
            messages=messages_proto,
            model=self.model,
            frequency_penalty=self.frequency_penalty,
            logit_bias=self.logit_bias,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            max_tokens=self.max_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            response_format=self.response_format.to_proto() if self.response_format else None,
            seed=self.seed,
            stop=stop_list,
            stream=self.stream,
            stream_options=self.stream_options.to_proto() if self.stream_options else None,
            temperature=self.temperature,
            top_p=self.top_p,
            user=self.user,
            regex=self.regex,
            min_tokens=self.min_tokens,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionRequest":
        messages = []
        for msg_param in proto.messages:
            if msg_param.HasField('generic_param'):
                content = msg_param.generic_param.content
                if content.HasField('text'):
                    msg_content = content.text
                else:
                    msg_content = [ChatCompletionMessageContentTextPart.from_proto(p)
                                 for p in content.parts]
                messages.append(ChatCompletionMessageGenericParam(
                    role=msg_param.generic_param.role,
                    content=msg_content
                ))
            else:  # user_param
                content = msg_param.user_param.content
                if content.HasField('text'):
                    msg_content = content.text
                else:
                    msg_content = []
                    msg_content.extend([ChatCompletionMessageContentTextPart.from_proto(p)
                                     for p in content.parts])
                    msg_content.extend([ChatCompletionMessageContentImagePart.from_proto(p)
                                     for p in content.images])
                messages.append(ChatCompletionMessageUserParam(
                    role=msg_param.user_param.role,
                    content=msg_content
                ))

        return cls(
            messages=messages,
            model=proto.model,
            frequency_penalty=proto.frequency_penalty,
            logit_bias=proto.logit_bias,
            logprobs=proto.logprobs,
            top_logprobs=proto.top_logprobs,
            max_tokens=proto.max_tokens,
            n=proto.n,
            presence_penalty=proto.presence_penalty,
            response_format=ResponseFormat.from_proto(proto.response_format) if proto.response_format else None,
            seed=proto.seed,
            stop=proto.stop,
            stream=proto.stream,
            stream_options=StreamOptions.from_proto(proto.stream_options) if proto.stream_options else None,
            temperature=proto.temperature,
            top_p=proto.top_p,
            user=proto.user,
            regex=proto.regex,
            min_tokens=proto.min_tokens,
            repetition_penalty=proto.repetition_penalty,
            stop_token_ids=proto.stop_token_ids,
            ignore_eos=proto.ignore_eos
        )


class ChatMessage(BaseModel, ProtoConvertible):
    role: Optional[str] = None
    content: Optional[str] = None

    def to_proto(self):
        return protocol_pb2.ChatMessage(
            role=self.role,
            content=self.content
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatMessage":
        return cls(
            role=proto.role,
            content=proto.content
        )


class ChatCompletionResponseChoice(BaseModel, ProtoConvertible):
    index: int
    message: ChatMessage
    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None
    finish_reason: str
    matched_stop: Union[None, int, str] = None

    def to_proto(self):
        logprobs_proto = None
        if self.logprobs:
            if isinstance(self.logprobs, LogProbs):
                logprobs_proto = protocol_pb2.LogProbsUnion(
                    basic=self.logprobs.to_proto()
                )
            elif isinstance(self.logprobs, ChoiceLogprobs):
                logprobs_proto = protocol_pb2.LogProbsUnion(
                    choice=self.logprobs.to_proto()
                )

        matched_stop = None
        if isinstance(self.matched_stop, int):
            matched_stop = protocol_pb2.MatchedStop(int_value=self.matched_stop)
        elif isinstance(self.matched_stop, str):
            matched_stop = protocol_pb2.MatchedStop(str_value=self.matched_stop)

        return protocol_pb2.ChatCompletionResponseChoice(
            index=self.index,
            message=self.message.to_proto(),
            logprobs=logprobs_proto,
            finish_reason=self.finish_reason,
            matched_stop=matched_stop
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionResponseChoice":
        logprobs = None
        if proto.logprobs:
            if proto.logprobs.basic:
                logprobs = LogProbs.from_proto(proto.logprobs.basic)
            elif proto.logprobs.choice:
                logprobs = ChoiceLogprobs.from_proto(proto.logprobs.choice)

        matched_stop = None
        if proto.matched_stop:
            if proto.matched_stop.int_value is not None:
                matched_stop = proto.matched_stop.int_value
            elif proto.matched_stop.str_value is not None:
                matched_stop = proto.matched_stop.str_value

        return cls(
            index=proto.index,
            message=ChatMessage.from_proto(proto.message),
            logprobs=logprobs,
            finish_reason=proto.finish_reason,
            matched_stop=matched_stop
        )


class ChatCompletionResponse(BaseModel, ProtoConvertible):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

    def to_proto(self):
        return protocol_pb2.ChatCompletionResponse(
            id=self.id,
            object=self.object,
            created=self.created,
            model=self.model,
            choices=[c.to_proto() for c in self.choices],
            usage=self.usage.to_proto()
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionResponse":
        return cls(
            id=proto.id,
            object=proto.object,
            created=proto.created,
            model=proto.model,
            choices=[ChatCompletionResponseChoice.from_proto(c) for c in proto.choices],
            usage=UsageInfo.from_proto(proto.usage)
        )


class DeltaMessage(BaseModel, ProtoConvertible):
    role: Optional[str] = None
    content: Optional[str] = None

    def to_proto(self):
        return protocol_pb2.DeltaMessage(
            role=self.role,
            content=self.content
        )

    @classmethod
    def from_proto(cls, proto) -> "DeltaMessage":
        return cls(
            role=proto.role,
            content=proto.content
        )


class ChatCompletionResponseStreamChoice(BaseModel, ProtoConvertible):
    index: int
    delta: DeltaMessage
    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None

    def to_proto(self):
        logprobs_proto = None
        if self.logprobs:
            if isinstance(self.logprobs, LogProbs):
                logprobs_proto = protocol_pb2.LogProbsUnion(basic=self.logprobs.to_proto())
            elif isinstance(self.logprobs, ChoiceLogprobs):
                logprobs_proto = protocol_pb2.LogProbsUnion(choice=self.logprobs.to_proto())

        matched_stop = None
        if isinstance(self.matched_stop, int):
            matched_stop = protocol_pb2.MatchedStop(int_value=self.matched_stop)
        elif isinstance(self.matched_stop, str):
            matched_stop = protocol_pb2.MatchedStop(str_value=self.matched_stop)

        return protocol_pb2.ChatCompletionResponseStreamChoice(
            index=self.index,
            delta=self.delta.to_proto(),
            logprobs=logprobs_proto,
            finish_reason=self.finish_reason,
            matched_stop=matched_stop
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionResponseStreamChoice":
        logprobs = None
        if proto.logprobs:
            assert bool(proto.logprobs.HasField('basic')) ^ bool(proto.logprobs.HasField('choice'))
            if proto.logprobs.HasField('basic'):
                logprobs = LogProbs.from_proto(proto.logprobs.basic)
            elif proto.logprobs.HasField('choice'):
                logprobs = ChoiceLogprobs.from_proto(proto.logprobs.choice)

        matched_stop = None
        if proto.matched_stop:
            if proto.matched_stop.int_value is not None:
                matched_stop = proto.matched_stop.int_value
            elif proto.matched_stop.str_value is not None:
                matched_stop = proto.matched_stop.str_value

        return cls(
            index=proto.index,
            delta=DeltaMessage.from_proto(proto.delta),
            logprobs=logprobs,
            finish_reason=proto.finish_reason,
            matched_stop=matched_stop
        )


class ChatCompletionStreamResponse(BaseModel, ProtoConvertible):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None

    def to_proto(self):
        return protocol_pb2.ChatCompletionStreamResponse(
            id=self.id,
            object=self.object,
            created=self.created,
            model=self.model,
            choices=[c.to_proto() for c in self.choices],
            usage=self.usage.to_proto() if self.usage else None
        )

    @classmethod
    def from_proto(cls, proto) -> "ChatCompletionStreamResponse":
        return cls(
            id=proto.id,
            object=proto.object,
            created=proto.created,
            model=proto.model,
            choices=[ChatCompletionResponseStreamChoice.from_proto(c) for c in proto.choices],
            usage=UsageInfo.from_proto(proto.usage) if proto.usage else None
        )


class EmbeddingRequest(BaseModel, ProtoConvertible):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings/create
    input: Union[List[int], List[List[int]], str, List[str]]
    model: str
    encoding_format: str = "float"
    dimensions: int = None
    user: Optional[str] = None

    def to_proto(self):
        input_content = protocol_pb2.EmbeddingInput()
        if isinstance(self.input, str):
            input_content.text = self.input
        elif isinstance(self.input, list):
            if all(isinstance(x, int) for x in self.input):
                input_content.tokens.extend(self.input)
            elif all(isinstance(x, list) for x in self.input):
                # Convert each inner list to a TokenSequence message
                for token_list in self.input:
                    token_seq = protocol_pb2.TokenSequence(tokens=token_list)
                    input_content.token_matrix.append(token_seq)
            else:
                input_content.texts.extend(self.input)

        return protocol_pb2.EmbeddingRequest(
            input=input_content,
            model=self.model,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            user=self.user
        )

    @classmethod
    def from_proto(cls, proto) -> "EmbeddingRequest":
        # Convert input from proto format
        if proto.input.text:
            input_value = proto.input.text
        elif proto.input.texts:
            input_value = proto.input.texts
        elif proto.input.tokens:
            input_value = proto.input.tokens
        else:
            input_value = [list(seq.tokens) for seq in proto.input.token_matrix]

        return cls(
            input=input_value,
            model=proto.model,
            encoding_format=proto.encoding_format,
            dimensions=proto.dimensions,
            user=proto.user
        )


class EmbeddingObject(BaseModel, ProtoConvertible):
    embedding: List[float]
    index: int
    object: str = "embedding"

    def to_proto(self):
        return protocol_pb2.EmbeddingObject(
            embedding=self.embedding,
            index=self.index,
            object=self.object
        )

    @classmethod
    def from_proto(cls, proto) -> "EmbeddingObject":
        return cls(
            embedding=proto.embedding,
            index=proto.index,
            object=proto.object
        )


class EmbeddingResponse(BaseModel, ProtoConvertible):
    data: List[EmbeddingObject]
    model: str
    object: str = "list"
    usage: Optional[UsageInfo] = None

    def to_proto(self):
        return protocol_pb2.EmbeddingResponse(
            data=[d.to_proto() for d in self.data],
            model=self.model,
            object=self.object,
            usage=self.usage.to_proto() if self.usage else None
        )

    @classmethod
    def from_proto(cls, proto) -> "EmbeddingResponse":
        return cls(
            data=[EmbeddingObject.from_proto(d) for d in proto.data],
            model=proto.model,
            object=proto.object,
            usage=UsageInfo.from_proto(proto.usage) if proto.usage else None
        )



