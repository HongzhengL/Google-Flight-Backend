"""Streaming response handlers for OpenAI-compatible endpoints."""

import json
import uuid
import logging
from typing import AsyncGenerator, List, Optional, Union

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from vllm import AsyncLLMEngine, SamplingParams
from vllm.outputs import RequestOutput

from app.models import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    CompletionRequest,
)

logger = logging.getLogger(__name__)


async def stream_chat_completion(
    request: ChatCompletionRequest,
    engine: AsyncLLMEngine
) -> StreamingResponse:
    """Generate streaming chat completion response.
    
    Parameters:
        request: OpenAI-compatible chat request
        engine: Initialized vLLM engine
        
    Returns:
        SSE StreamingResponse with token-by-token output
        
    Error handling:
        Graceful stream termination on errors
    """
    try:
        # Convert messages to prompt
        prompt = _messages_to_prompt(request.messages)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            seed=request.seed,
            n=request.n,
        )
        
        # Generate request ID
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        # Create generator for streaming
        async def generate_stream() -> AsyncGenerator[str, None]:
            try:
                # Start generation
                results_generator = engine.generate(prompt, sampling_params, request_id)
                
                # Stream tokens
                async for request_output in results_generator:
                    if request_output.finished:
                        # Send final chunk with finish reason
                        for i, output in enumerate(request_output.outputs):
                            chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                model=request.model,
                                choices=[
                                    ChatCompletionStreamResponseChoice(
                                        index=i,
                                        delta=ChatCompletionStreamResponseDelta(),
                                        finish_reason=_get_finish_reason(output.finish_reason),
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                        
                        # Send [DONE] marker
                        yield "data: [DONE]\n\n"
                    else:
                        # Stream intermediate tokens
                        for i, output in enumerate(request_output.outputs):
                            # Get new tokens since last update
                            new_text = output.text[len(output.text) - len(output.token_ids):]
                            if new_text:
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamResponseChoice(
                                            index=i,
                                            delta=ChatCompletionStreamResponseDelta(
                                                content=new_text,
                                                role="assistant" if output.text == new_text else None,
                                            ),
                                        )
                                    ],
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                                
            except Exception as e:
                logger.error(f"Streaming generation error: {str(e)}")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "generation_error",
                        "code": "stream_generation_failed"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable proxy buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create streaming response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_completion(
    request: CompletionRequest,
    engine: AsyncLLMEngine
) -> StreamingResponse:
    """Generate streaming completion response.
    
    Parameters:
        request: OpenAI-compatible completion request
        engine: Initialized vLLM engine
        
    Returns:
        SSE StreamingResponse with token-by-token output
    """
    try:
        # Handle different prompt formats
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        elif isinstance(request.prompt, list):
            if all(isinstance(p, str) for p in request.prompt):
                prompts = request.prompt
            else:
                # Token IDs not directly supported, convert or raise error
                raise HTTPException(
                    status_code=400,
                    detail="Token ID prompts not supported in streaming mode"
                )
        else:
            prompts = [str(request.prompt)]
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            seed=request.seed,
            n=request.n,
            logprobs=request.logprobs,
        )
        
        # Generate request ID
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        
        # Create generator for streaming
        async def generate_stream() -> AsyncGenerator[str, None]:
            try:
                for prompt_idx, prompt in enumerate(prompts):
                    # Start generation for this prompt
                    results_generator = engine.generate(prompt, sampling_params, f"{request_id}-{prompt_idx}")
                    
                    # Stream tokens
                    async for request_output in results_generator:
                        if request_output.finished:
                            # Send final chunk with finish reason
                            for i, output in enumerate(request_output.outputs):
                                chunk = {
                                    "id": request_id,
                                    "object": "text_completion",
                                    "created": int(uuid.uuid4().time),
                                    "model": request.model,
                                    "choices": [{
                                        "text": "",
                                        "index": prompt_idx * request.n + i,
                                        "finish_reason": _get_finish_reason(output.finish_reason),
                                        "logprobs": None,  # TODO: Add logprobs support
                                    }]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                        else:
                            # Stream intermediate tokens
                            for i, output in enumerate(request_output.outputs):
                                # Get new tokens since last update
                                new_text = output.text[len(output.text) - len(output.token_ids):]
                                if new_text:
                                    chunk = {
                                        "id": request_id,
                                        "object": "text_completion",
                                        "created": int(uuid.uuid4().time),
                                        "model": request.model,
                                        "choices": [{
                                            "text": new_text,
                                            "index": prompt_idx * request.n + i,
                                            "finish_reason": None,
                                            "logprobs": None,  # TODO: Add logprobs support
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send [DONE] marker
                yield "data: [DONE]\n\n"
                                    
            except Exception as e:
                logger.error(f"Streaming generation error: {str(e)}")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "generation_error",
                        "code": "stream_generation_failed"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create streaming response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _messages_to_prompt(messages: List) -> str:
    """Convert chat messages to a single prompt string.
    
    Parameters:
        messages: List of chat messages
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    for message in messages:
        role = message.role
        content = message.content or ""
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        else:
            # Handle other roles as needed
            prompt_parts.append(f"{role.capitalize()}: {content}")
    
    # Add final assistant prompt marker
    prompt_parts.append("Assistant:")
    
    return "\n\n".join(prompt_parts)


def _get_finish_reason(vllm_finish_reason: Optional[str]) -> Optional[str]:
    """Convert vLLM finish reason to OpenAI format.
    
    Parameters:
        vllm_finish_reason: vLLM's finish reason
        
    Returns:
        OpenAI-compatible finish reason
    """
    if not vllm_finish_reason:
        return None
    
    # Map vLLM reasons to OpenAI reasons
    reason_map = {
        "stop": "stop",
        "length": "length",
        "abort": "stop",
        "error": "stop",
    }
    
    return reason_map.get(vllm_finish_reason, "stop")