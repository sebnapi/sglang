import time

from http import HTTPStatus
from typing import Dict, List

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import ValidationError

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.openai_api.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ErrorResponse,
    UsageInfo,
)

def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
):
    return ErrorResponse(
        error={"message": message, "type": err_type, "code": HTTPStatus.BAD_REQUEST}
    )

def v1_generate_request(
    all_requests: List[CompletionRequest], request_ids: List[str] = None
):
    prompts = []
    sampling_params_list = []
    return_logprobs = []
    logprob_start_lens = []
    top_logprobs_nums = []

    # NOTE: with openai API, the prompt's logprobs are always not computed
    first_prompt_type = type(all_requests[0].prompt)
    for request in all_requests:
        assert (
            type(request.prompt) is first_prompt_type
        ), "All prompts must be of the same type in file input settings"
        if len(all_requests) > 1 and request.n > 1:
            raise ValueError(
                "Parallel sampling is not supported for completions from files"
            )
        if request.echo and request.logprobs:
            logger.warning(
                "Echo is not compatible with logprobs. "
                "To compute logprobs of input prompt, please use SGLang /request API."
            )

    for request in all_requests:
        prompts.append(request.prompt)
        return_logprobs.append(request.logprobs is not None and request.logprobs > 0)
        logprob_start_lens.append(-1)
        top_logprobs_nums.append(
            request.logprobs if request.logprobs is not None else 0
        )
        sampling_params = []
        if isinstance(request.no_stop_trim, list):
            num_reqs = len(request.prompt)
        else:
            num_reqs = 1
        for i in range(num_reqs):
            sampling_params.append(
                {
                    "temperature": request.temperature,
                    "max_new_tokens": request.max_tokens,
                    "min_new_tokens": request.min_tokens,
                    "stop": request.stop,
                    "stop_token_ids": request.stop_token_ids,
                    "top_p": request.top_p,
                    "presence_penalty": request.presence_penalty,
                    "frequency_penalty": request.frequency_penalty,
                    "repetition_penalty": request.repetition_penalty,
                    "regex": request.regex,
                    "json_schema": request.json_schema,
                    "n": request.n,
                    "ignore_eos": request.ignore_eos,
                    "no_stop_trim": (
                        request.no_stop_trim
                        if not isinstance(request.no_stop_trim, list)
                        else request.no_stop_trim[i]
                    ),
                }
            )
        if num_reqs == 1:
            sampling_params_list.append(sampling_params[0])
        else:
            sampling_params_list.append(sampling_params)

    if len(all_requests) == 1:
        prompt = prompts[0]
        sampling_params_list = sampling_params_list[0]
        logprob_start_lens = logprob_start_lens[0]
        return_logprobs = return_logprobs[0]
        top_logprobs_nums = top_logprobs_nums[0]
        if isinstance(prompt, str) or isinstance(prompt[0], str):
            prompt_kwargs = {"text": prompt}
        else:
            prompt_kwargs = {"input_ids": prompt}
    else:
        if isinstance(prompts[0], str):
            prompt_kwargs = {"text": prompts}
        else:
            prompt_kwargs = {"input_ids": prompts}

    adapted_request = GenerateReqInput(
        **prompt_kwargs,
        sampling_params=sampling_params_list,
        return_logprob=return_logprobs,
        top_logprobs_num=top_logprobs_nums,
        logprob_start_len=logprob_start_lens,
        return_text_in_logprobs=True,
        stream=all_requests[0].stream,
        rid=request_ids,
    )

    if len(all_requests) == 1:
        return adapted_request, all_requests[0]
    return adapted_request, all_requests


def v1_generate_response(request, ret, tokenizer_manager, to_file=False):
    choices = []
    echo = False

    if (not isinstance(request, list)) and request.echo:
        # TODO: handle the case propmt is token ids
        if isinstance(request.prompt, list) and isinstance(request.prompt[0], str):
            # for the case of multiple str prompts
            prompts = request.prompt
        elif isinstance(request.prompt, list) and isinstance(request.prompt[0], list):
            # for the case of multiple token ids prompts
            prompts = [
                tokenizer_manager.tokenizer.decode(prompt, skip_special_tokens=True)
                for prompt in request.prompt
            ]
        elif isinstance(request.prompt, list) and isinstance(request.prompt[0], int):
            # for the case of single token ids prompt
            prompts = [
                tokenizer_manager.tokenizer.decode(
                    request.prompt, skip_special_tokens=True
                )
            ]
        else:
            # for the case of single str prompt
            prompts = [request.prompt]
        echo = True

    for idx, ret_item in enumerate(ret):
        text = ret_item["text"]
        if isinstance(request, list) and request[idx].echo:
            echo = True
            text = request[idx].prompt + text
        if (not isinstance(request, list)) and echo:
            prompt_index = idx // request.n
            text = prompts[prompt_index] + text

        logprobs = False
        if isinstance(request, list) and request[idx].logprobs:
            logprobs = True
        elif (not isinstance(request, list)) and request.logprobs:
            logprobs = True
        if logprobs:
            if echo:
                input_token_logprobs = ret_item["meta_info"]["input_token_logprobs"]
                input_top_logprobs = ret_item["meta_info"]["input_top_logprobs"]
            else:
                input_token_logprobs = None
                input_top_logprobs = None

            logprobs = to_openai_style_logprobs(
                input_token_logprobs=input_token_logprobs,
                input_top_logprobs=input_top_logprobs,
                output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
                output_top_logprobs=ret_item["meta_info"]["output_top_logprobs"],
            )
        else:
            logprobs = None

        finish_reason = ret_item["meta_info"]["finish_reason"]

        if to_file:
            # to make the choise data json serializable
            choice_data = {
                "index": 0,
                "text": text,
                "logprobs": logprobs,
                "finish_reason": (finish_reason["type"] if finish_reason else ""),
                "matched_stop": (
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
            }
        else:
            choice_data = CompletionResponseChoice(
                index=idx,
                text=text,
                logprobs=logprobs,
                finish_reason=(finish_reason["type"] if finish_reason else ""),
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
            )

        choices.append(choice_data)

    if to_file:
        responses = []
        for i, choice in enumerate(choices):
            response = {
                "status_code": 200,
                "request_id": ret[i]["meta_info"]["id"],
                "body": {
                    # remain the same but if needed we can change that
                    "id": ret[i]["meta_info"]["id"],
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request[i].model,
                    "choices": choice,
                    "usage": {
                        "prompt_tokens": ret[i]["meta_info"]["prompt_tokens"],
                        "completion_tokens": ret[i]["meta_info"]["completion_tokens"],
                        "total_tokens": ret[i]["meta_info"]["prompt_tokens"]
                        + ret[i]["meta_info"]["completion_tokens"],
                    },
                    "system_fingerprint": None,
                },
            }
            responses.append(response)
        return responses
    else:
        prompt_tokens = sum(
            ret[i]["meta_info"]["prompt_tokens"] for i in range(0, len(ret), request.n)
        )
        completion_tokens = sum(item["meta_info"]["completion_tokens"] for item in ret)
        response = CompletionResponse(
            id=ret[0]["meta_info"]["id"],
            model=request.model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    return response

async def v1_completions(tokenizer_manager, raw_request: Request):
    print('MY v1_completions *** '*10)

    request_json = await raw_request.json()
    all_requests = [CompletionRequest(**request_json)]
    adapted_request, request = v1_generate_request(all_requests)

    # Non-streaming response
    try:
        ret = await tokenizer_manager.generate_request(
            adapted_request, raw_request
        ).__anext__()
    except ValueError as e:
        return create_error_response(str(e))
    if not isinstance(ret, list):
        ret = [ret]

    response = v1_generate_response(request, ret, tokenizer_manager)
    return response
