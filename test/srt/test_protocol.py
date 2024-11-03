import pytest
import time
import math
from typing import List, Dict, Any, Callable
import numpy as np
import functools

from sglang.srt.openai_api.protocol import (
    ModelCard,
    ModelList,
    ErrorResponse,
    LogProbs,
    TopLogprob,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    UsageInfo,
    StreamOptions,
    JsonSchemaResponseFormat,
    FileRequest,
    FileResponse,
    FileDeleteResponse,
    BatchMetadata,
    BatchRequest,
    BatchError,
    BatchResponse,
    StopTrimConfig,
    CompletionRequest,
    CompletionResponseChoice,
    CompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    DeltaMessage,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    EmbeddingRequest,
    EmbeddingObject,
    EmbeddingResponse,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageGenericParam,
    ChatCompletionMessageUserParam,
    ResponseFormat,
    ChatCompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)

def test_model_card():
    # Create a model card
    card = ModelCard(
        id="test-model",
        object="model",
        created=1234567890,
        owned_by="test-owner",
        root="test-root"
    )

    # Convert to proto and back
    proto = card.to_proto()
    card2 = ModelCard.from_proto(proto)

    # Verify fields match
    assert card.id == card2.id
    assert card.object == card2.object
    assert card.created == card2.created
    assert card.owned_by == card2.owned_by
    assert card.root == card2.root

def test_model_list():
    # Create model cards
    cards = [
        ModelCard(id="model1", created=1234567890),
        ModelCard(id="model2", created=1234567891),
    ]

    # Create model list
    model_list = ModelList(data=cards)

    # Convert to proto and back
    proto = model_list.to_proto()
    model_list2 = ModelList.from_proto(proto)

    # Verify fields match
    assert len(model_list.data) == len(model_list2.data)
    for card1, card2 in zip(model_list.data, model_list2.data):
        assert card1.id == card2.id
        assert card1.created == card2.created

def test_error_response():
    error = ErrorResponse(
        message="Test error",
        type="test_error",
        param="test_param",
        code=400
    )

    proto = error.to_proto()
    error2 = ErrorResponse.from_proto(proto)

    assert error.message == error2.message
    assert error.type == error2.type
    assert error.param == error2.param
    assert error.code == error2.code

def test_log_probs():
    log_probs = LogProbs(
        text_offset=[0, 4, 8],
        token_logprobs=[-0.5, -1.0, -1.5],
        tokens=["test", "tokens", "here"],
        top_logprobs=[{"token1": -0.5, "token2": -1.0}, None, {"token3": -1.5}]
    )

    proto = log_probs.to_proto()
    log_probs2 = LogProbs.from_proto(proto)

    assert log_probs.text_offset == log_probs2.text_offset
    assert log_probs.token_logprobs == log_probs2.token_logprobs
    assert log_probs.tokens == log_probs2.tokens
    assert log_probs.top_logprobs == log_probs2.top_logprobs

def test_top_logprob():
    top_logprob = TopLogprob(
        token="test",
        logprob=-0.5,
        bytes=[10]
    )

    proto = top_logprob.to_proto()
    top_logprob2 = TopLogprob.from_proto(proto)

    assert top_logprob.token == top_logprob2.token
    assert top_logprob.logprob == top_logprob2.logprob
    assert top_logprob.bytes == top_logprob2.bytes

def test_chat_completion_token_logprob():
    token_logprob = ChatCompletionTokenLogprob(
        token="test",
        logprob=-0.5,
        bytes=[5],
        top_logprobs=[TopLogprob(token="alt", logprob=-1.0, bytes=[5])]
    )

    proto = token_logprob.to_proto()
    token_logprob2 = ChatCompletionTokenLogprob.from_proto(proto)

    assert token_logprob.token == token_logprob2.token
    assert token_logprob.logprob == token_logprob2.logprob
    assert len(token_logprob.top_logprobs) == len(token_logprob2.top_logprobs)
    assert token_logprob.top_logprobs[0].token == token_logprob2.top_logprobs[0].token

def test_choice_logprobs():
    choice_logprobs = ChoiceLogprobs(
        content=[ChatCompletionTokenLogprob(
            token="test",
            logprob=-0.5,
            bytes=[5],
            top_logprobs=[TopLogprob(token="alt", logprob=-1.0, bytes=[5])]
        )]
    )

    proto = choice_logprobs.to_proto()
    choice_logprobs2 = ChoiceLogprobs.from_proto(proto)

    assert len(choice_logprobs.content) == len(choice_logprobs2.content)
    assert choice_logprobs.content[0].token == choice_logprobs2.content[0].token

def test_chat_completion_message_content_text_part():
    text_part = ChatCompletionMessageContentTextPart(
        type="text",
        text="Hello world"
    )

    proto = text_part.to_proto()
    text_part2 = ChatCompletionMessageContentTextPart.from_proto(proto)

    assert text_part.type == text_part2.type
    assert text_part.text == text_part2.text

def test_chat_completion_message_content_image_url():
    image_url = ChatCompletionMessageContentImageURL(
        url="http://example.com/image.jpg"
    )

    proto = image_url.to_proto()
    image_url2 = ChatCompletionMessageContentImageURL.from_proto(proto)

    assert image_url.url == image_url2.url

def test_chat_completion_message_content_image_part():
    image_part = ChatCompletionMessageContentImagePart(
        type="image_url",
        image_url=ChatCompletionMessageContentImageURL(url="http://example.com/image.jpg")
    )

    proto = image_part.to_proto()
    image_part2 = ChatCompletionMessageContentImagePart.from_proto(proto)

    assert image_part.type == image_part2.type
    assert image_part.image_url.url == image_part2.image_url.url

def test_chat_completion_message_generic_param():
    generic_param = ChatCompletionMessageGenericParam(
        role="assistant",
        content="Hello"
    )

    proto = generic_param.to_proto()
    generic_param2 = ChatCompletionMessageGenericParam.from_proto(proto)

    assert generic_param.role == generic_param2.role
    assert generic_param.content == generic_param2.content

def test_chat_completion_message_user_param():
    user_param = ChatCompletionMessageUserParam(
        role="user",
        content=[
            ChatCompletionMessageContentTextPart(type="text", text="Hello"),
            ChatCompletionMessageContentImagePart(
                type="image_url",
                image_url=ChatCompletionMessageContentImageURL(url="http://example.com/image.jpg")
            )
        ]
    )

    proto = user_param.to_proto()
    user_param2 = ChatCompletionMessageUserParam.from_proto(proto)

    assert user_param.role == user_param2.role
    assert len(user_param.content) == len(user_param2.content)
    assert user_param.content[0].text == user_param2.content[0].text

def test_response_format():
    response_format = ResponseFormat(
        type="json_object",
        json_schema=JsonSchemaResponseFormat(name="test", schema={"type": "object"})
    )

    proto = response_format.to_proto()
    response_format2 = ResponseFormat.from_proto(proto)

    assert response_format.type == response_format2.type
    assert response_format.json_schema.name == response_format2.json_schema.name
    assert response_format.json_schema.schema == response_format2.json_schema.schema
    assert response_format.json_schema.strict == response_format2.json_schema.strict

def test_chat_completion_request():
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[
            ChatCompletionMessageUserParam(
                role="user",
                content=[ChatCompletionMessageContentTextPart(type="text", text="Hello")]
            )
        ],
        temperature=0.7,
        response_format=ResponseFormat(type="json_object", json_schema=JsonSchemaResponseFormat(name="test", schema={"type": "object"}))
    )

    proto = request.to_proto()
    request2 = ChatCompletionRequest.from_proto(proto)

    assert request.model == request2.model
    assert len(request.messages) == len(request2.messages)
    assert math.isclose(request.temperature, request2.temperature, rel_tol=1e-6)
    assert request.response_format.type == request2.response_format.type

def test_completion_response_stream_choice():
    choice = CompletionResponseStreamChoice(
        index=0,
        text="Hello",
        logprobs=LogProbs(
            text_offset=[0],
            token_logprobs=[-0.5],
            tokens=["Hello"],
            top_logprobs=[{"Hello": -0.5, "Hi": -1.0}]
        ),
        finish_reason="stop"
    )

    proto = choice.to_proto()
    choice2 = CompletionResponseStreamChoice.from_proto(proto)

    assert choice.index == choice2.index
    assert choice.text == choice2.text
    assert choice.finish_reason == choice2.finish_reason
    assert choice.logprobs.text_offset == choice2.logprobs.text_offset
    assert choice.logprobs.token_logprobs == choice2.logprobs.token_logprobs
    assert choice.logprobs.tokens == choice2.logprobs.tokens
    assert choice.logprobs.top_logprobs == choice2.logprobs.top_logprobs

def test_completion_stream_response():
    response = CompletionStreamResponse(
        id="test-stream",
        object="text_completion",
        created=int(time.time()),
        model="gpt-4",
        choices=[
            CompletionResponseStreamChoice(
                index=0,
                text="Hello",
                logprobs=None,
                finish_reason="stop"
            )
        ]
    )

    proto = response.to_proto()
    response2 = CompletionStreamResponse.from_proto(proto)

    assert response.id == response2.id
    assert response.model == response2.model
    assert len(response.choices) == len(response2.choices)
    assert response.choices[0].text == response2.choices[0].text

def test_usage_info():
    usage = UsageInfo(
        prompt_tokens=10,
        total_tokens=20,
        completion_tokens=10,
        prompt_tokens_details={"cached": 5, "uncached": 5}
    )

    proto = usage.to_proto()
    usage2 = UsageInfo.from_proto(proto)

    assert usage.prompt_tokens == usage2.prompt_tokens
    assert usage.total_tokens == usage2.total_tokens
    assert usage.completion_tokens == usage2.completion_tokens
    assert usage.prompt_tokens_details == usage2.prompt_tokens_details

def test_stream_options():
    options = StreamOptions(
        include_usage=True
    )

    proto = options.to_proto()
    options2 = StreamOptions.from_proto(proto)

    assert options.include_usage == options2.include_usage

def test_json_schema_response_format():
    format = JsonSchemaResponseFormat(
        name="test_format",
        schema_={
            'type': 'object',
            'properties': {
                'a': {'type': 'string'}
            }
        },
        strict=False
    )

    proto = format.to_proto()
    format2 = JsonSchemaResponseFormat.from_proto(proto)

    assert format.name == format2.name
    assert format.schema_ == format2.schema_
    assert format.strict == format2.strict

def test_file_request():
    request = FileRequest(
        file=b"test_content",
        purpose="fine-tune"
    )

    proto = request.to_proto()
    request2 = FileRequest.from_proto(proto)

    assert request.file == request2.file
    assert request.purpose == request2.purpose

def test_file_response():
    response = FileResponse(
        id="file-123",
        object="file",
        bytes=1000,
        created_at=int(time.time()),
        filename="test.txt",
        purpose="fine-tune"
    )

    proto = response.to_proto()
    response2 = FileResponse.from_proto(proto)

    assert response.id == response2.id
    assert response.filename == response2.filename
    assert response.bytes == response2.bytes
    assert response.created_at == response2.created_at

def test_file_delete_response():
    response = FileDeleteResponse(
        id="file-123",
        deleted=True
    )

    proto = response.to_proto()
    response2 = FileDeleteResponse.from_proto(proto)

    assert response.id == response2.id
    assert response.deleted == response2.deleted

def test_batch_error():
    error = BatchError(
        message="Test error",
        code="invalid_request"
    )

    proto = error.to_proto()
    error2 = BatchError.from_proto(proto)

    assert error.message == error2.message
    assert error.code == error2.code

def test_stop_trim_config():
    config = StopTrimConfig(
        single=True,
        multiple=[True, False]
    )

    proto = config.to_proto()
    config2 = StopTrimConfig.from_proto(proto)

    assert config.single == config2.single
    assert config.multiple == config2.multiple

def test_completion_request():
    request = CompletionRequest(
        model="test-model",
        prompt="Test prompt",
        max_tokens=100,
        temperature=0.7,
        stop=[".", "\n"],
        stream=True
    )

    proto = request.to_proto()
    request2 = CompletionRequest.from_proto(proto)

    assert request.model == request2.model
    assert request.prompt == request2.prompt
    assert request.max_tokens == request2.max_tokens
    assert np.isclose(request.temperature, request2.temperature)
    assert request.stop == request2.stop
    assert request.stream == request2.stream

def test_completion_response():
    choice = CompletionResponseChoice(
        index=0,
        text="Test completion",
        logprobs=LogProbs(tokens=["test", "completion"]),
        finish_reason="length"
    )

    response = CompletionResponse(
        id="test-completion",
        model="test-model",
        choices=[choice],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )

    proto = response.to_proto()
    response2 = CompletionResponse.from_proto(proto)

    assert response.id == response2.id
    assert response.model == response2.model
    assert len(response.choices) == len(response2.choices)
    assert response.choices[0].text == response2.choices[0].text
    assert response.usage.total_tokens == response2.usage.total_tokens

def test_chat_completion_response():
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="Test response"),
        finish_reason="stop"
    )

    response = ChatCompletionResponse(
        id="test-chat",
        model="test-model",
        choices=[choice],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )

    proto = response.to_proto()
    response2 = ChatCompletionResponse.from_proto(proto)

    assert response.id == response2.id
    assert response.model == response2.model
    assert len(response.choices) == len(response2.choices)
    assert response.choices[0].message.content == response2.choices[0].message.content
    assert response.usage.total_tokens == response2.usage.total_tokens

def test_chat_completion_stream_response():
    choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content="Test"),
        finish_reason=None,
        logprobs=LogProbs(
            text_offset=[0],
            token_logprobs=[0.5],
            tokens=["test"],
            top_logprobs=[{"test": 0.5}]
        )
        )

    response = ChatCompletionStreamResponse(
        id="test-chat-stream",
        model="test-model",
        choices=[choice]
    )

    proto = response.to_proto()
    response2 = ChatCompletionStreamResponse.from_proto(proto)

    assert response.id == response2.id
    assert response.model == response2.model
    assert len(response.choices) == len(response2.choices)
    assert response.choices[0].delta.content == response2.choices[0].delta.content
    assert response.choices[0].logprobs.tokens == response2.choices[0].logprobs.tokens

def test_embedding_request():
    request = EmbeddingRequest(
        model="test-model",
        input="Test input",
        encoding_format="float"
    )

    proto = request.to_proto()
    request2 = EmbeddingRequest.from_proto(proto)

    assert request.model == request2.model
    assert request.input == request2.input
    assert request.encoding_format == request2.encoding_format

def test_embedding_response():
    embedding_obj = EmbeddingObject(
        embedding=[0.1, 0.2, 0.3],
        index=0
    )

    response = EmbeddingResponse(
        data=[embedding_obj],
        model="test-model",
        usage=UsageInfo(prompt_tokens=10, total_tokens=10)
    )

    proto = response.to_proto()
    response2 = EmbeddingResponse.from_proto(proto)

    assert response.model == response2.model
    assert len(response.data) == len(response2.data)
    assert np.allclose(response.data[0].embedding, response2.data[0].embedding)
    assert response.usage.total_tokens == response2.usage.total_tokens

def test_batch_request():
    metadata = [BatchMetadata(key="test_key", value="test_value")]
    request = BatchRequest(
        input_file_id="test-file",
        endpoint="/v1/completions",
        completion_window="1h",
        metadata=metadata
    )

    proto = request.to_proto()
    request2 = BatchRequest.from_proto(proto)

    assert request.input_file_id == request2.input_file_id
    assert request.endpoint == request2.endpoint
    assert request.completion_window == request2.completion_window
    assert request.metadata[0].key == request2.metadata[0].key
    assert request.metadata[0].value == request2.metadata[0].value

def test_batch_response():
    metadata = [BatchMetadata(key="test_key", value="test_value")]
    response = BatchResponse(
        id="test-batch",
        endpoint="/v1/completions",
        input_file_id="test-file",
        completion_window="1h",
        created_at=int(time.time()),
        metadata=metadata
    )

    proto = response.to_proto()
    response2 = BatchResponse.from_proto(proto)

    assert response.id == response2.id
    assert response.endpoint == response2.endpoint
    assert response.input_file_id == response2.input_file_id
    assert response.completion_window == response2.completion_window
    assert response.metadata[0].key == response2.metadata[0].key
    assert response.metadata[0].value == response2.metadata[0].value


def _test_completion_request_prompt_union(test_value):
    request = CompletionRequest(
        model="test-model",
        prompt=test_value
    )
    proto = request.to_proto()
    request2 = CompletionRequest.from_proto(proto)
    assert request2.prompt == test_value

def test_completion_request_prompt_union_list_int():
    _test_completion_request_prompt_union([1, 2, 3])

def test_completion_request_prompt_union_list_list_int():
    _test_completion_request_prompt_union([[1, 2], [3, 4]])

def test_completion_request_prompt_union_str():
    _test_completion_request_prompt_union("test prompt")

def test_completion_request_prompt_union_list_str():
    _test_completion_request_prompt_union(["test1", "test2"])

def _test_completion_request_stop_union(test_value):
    request = CompletionRequest(
        model="test-model",
        prompt="test",
        stop=test_value
    )
    proto = request.to_proto()
    request2 = CompletionRequest.from_proto(proto)
    # For single string, it gets converted to list
    expected = [test_value] if isinstance(test_value, str) else test_value
    assert request2.stop == expected

def test_completion_request_stop_union_str():
    _test_completion_request_stop_union("stop")

def test_completion_request_stop_union_list_str():
    _test_completion_request_stop_union(["stop1", "stop2"])

def _test_completion_request_no_stop_trim_union(test_value):
    request = CompletionRequest(
        model="test-model",
        prompt="test",
        no_stop_trim=test_value
    )
    proto = request.to_proto()
    request2 = CompletionRequest.from_proto(proto)
    assert request2.no_stop_trim == test_value

def test_completion_request_no_stop_trim_union_bool():
    _test_completion_request_no_stop_trim_union(True)

def test_completion_request_no_stop_trim_union_list_bool():
    _test_completion_request_no_stop_trim_union([True, False])

def _test_completion_response_choice_matched_stop_union(test_value):
    choice = CompletionResponseChoice(
        index=0,
        text="test",
        matched_stop=test_value
    )
    proto = choice.to_proto()
    choice2 = CompletionResponseChoice.from_proto(proto)
    assert choice2.matched_stop == test_value

def test_completion_response_choice_matched_stop_union_none():
    _test_completion_response_choice_matched_stop_union(None)

def test_completion_response_choice_matched_stop_union_int():
    _test_completion_response_choice_matched_stop_union(1)

def test_completion_response_choice_matched_stop_union_str():
    _test_completion_response_choice_matched_stop_union("stop")

def _test_completion_response_stream_choice_logprobs_union(test_value):
    choice = CompletionResponseStreamChoice(
        index=0,
        text="test",
        logprobs=test_value
    )
    proto = choice.to_proto()
    choice2 = CompletionResponseStreamChoice.from_proto(proto)
    assert choice2.logprobs == test_value

def test_completion_response_stream_choice_logprobs_union_logprobs():
    _test_completion_response_stream_choice_logprobs_union(LogProbs(tokens=["test"]))

def test_chat_completion_response_stream_choice_logprobs_union_choice_logprobs():
    choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content="test"),
        logprobs=ChoiceLogprobs(content=[
            ChatCompletionTokenLogprob(token="test", logprob=-0.5, bytes=[5], top_logprobs=[])
        ])
    )
    proto = choice.to_proto()
    choice2 = ChatCompletionResponseStreamChoice.from_proto(proto)
    assert choice2.logprobs == choice.logprobs

def test_chat_completion_response_stream_choice_logprobs_union_logprobs():
    choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content="test"),
        logprobs=LogProbs(
            text_offset=[0],
            token_logprobs=[-0.5],
            tokens=["test"],
            top_logprobs=[]
        )
    )
    proto = choice.to_proto()
    choice2 = ChatCompletionResponseStreamChoice.from_proto(proto)
    assert choice2.logprobs == choice.logprobs


def _test_chat_completion_message_generic_param_content_union(test_value):
    param = ChatCompletionMessageGenericParam(
        role="system",
        content=test_value
    )
    proto = param.to_proto()
    param2 = ChatCompletionMessageGenericParam.from_proto(proto)
    assert param2.content == test_value

def test_chat_completion_message_generic_param_content_union_str():
    _test_chat_completion_message_generic_param_content_union("test content")

def test_chat_completion_message_generic_param_content_union_list_text_part():
    _test_chat_completion_message_generic_param_content_union([
        ChatCompletionMessageContentTextPart(type="text", text="test")
    ])

def _test_chat_completion_message_user_param_content_union(test_value):
    param = ChatCompletionMessageUserParam(
        role="user",
        content=test_value
    )
    proto = param.to_proto()
    param2 = ChatCompletionMessageUserParam.from_proto(proto)
    assert param2.content == test_value

def test_chat_completion_message_user_param_content_union_str():
    _test_chat_completion_message_user_param_content_union("test content")

def test_chat_completion_message_user_param_content_union_list_content_parts():
    _test_chat_completion_message_user_param_content_union([
        ChatCompletionMessageContentTextPart(type="text", text="test"),
        ChatCompletionMessageContentImagePart(
            type="image_url",
            image_url=ChatCompletionMessageContentImageURL(url="http://test.com")
        )
    ])

def _test_embedding_request_input_union(test_value):
    request = EmbeddingRequest(
        model="test-model",
        input=test_value
    )
    proto = request.to_proto()
    request2 = EmbeddingRequest.from_proto(proto)
    assert request2.input == test_value

def test_embedding_request_input_union_list_int():
    _test_embedding_request_input_union([1, 2, 3])

def test_embedding_request_input_union_list_list_int():
    _test_embedding_request_input_union([[1, 2], [3, 4]])

def test_embedding_request_input_union_str():
    _test_embedding_request_input_union("test input")

def test_embedding_request_input_union_list_str():
    _test_embedding_request_input_union(["test1", "test2"])

if __name__ == "__main__":
    pytest.main([__file__])
