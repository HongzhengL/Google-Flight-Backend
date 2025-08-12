"""Main FastAPI application for GPT-OSS-20B API service."""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from vllm import AsyncLLMEngine, SamplingParams

from app.config import settings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ModelListResponse,
    ModelObject,
    ErrorResponse,
)
from app.llm_engine import engine_manager
from app.streaming import stream_chat_completion, stream_completion
from app.health import health_check, readiness_check, liveness_check

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    try:
        await engine_manager.initialize_engine()
        logger.info("Model loaded and engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {str(e)}")
        # Continue running even if model fails to load initially
        # This allows health checks to work and debugging
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await engine_manager.shutdown()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Dependency to get engine
async def get_engine() -> AsyncLLMEngine:
    """Get the initialized vLLM engine."""
    try:
        return engine_manager.get_engine()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# Health check endpoints
@app.get("/health")
async def health_endpoint():
    """Basic health check endpoint."""
    return await health_check()


@app.get("/ready")
async def readiness_endpoint():
    """Readiness probe endpoint."""
    try:
        engine = engine_manager.get_engine()
    except RuntimeError:
        engine = None
    return await readiness_check(engine)


@app.get("/livez")
async def liveness_endpoint():
    """Liveness probe endpoint."""
    return await liveness_check()


# OpenAI-compatible endpoints
@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    engine: AsyncLLMEngine = Depends(get_engine)
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Handle streaming request
        if request.stream:
            return await stream_chat_completion(request, engine)
        
        # Non-streaming request
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
        
        # Generate completion
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        results = await engine.generate(prompt, sampling_params, request_id)
        
        # Format response
        choices = []
        total_tokens = 0
        for i, output in enumerate(results.outputs):
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=output.text),
                    finish_reason=_get_finish_reason(output.finish_reason),
                )
            )
            total_tokens += len(output.token_ids)
        
        return ChatCompletionResponse(
            id=request_id,
            model=request.model,
            choices=choices,
            usage={
                "prompt_tokens": len(results.prompt_token_ids),
                "completion_tokens": total_tokens,
                "total_tokens": len(results.prompt_token_ids) + total_tokens,
            },
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    engine: AsyncLLMEngine = Depends(get_engine)
) -> CompletionResponse | StreamingResponse:
    """OpenAI-compatible completions endpoint."""
    try:
        # Handle streaming request
        if request.stream:
            return await stream_completion(request, engine)
        
        # Non-streaming request
        # Handle different prompt formats
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        elif isinstance(request.prompt, list):
            if all(isinstance(p, str) for p in request.prompt):
                prompts = request.prompt
            else:
                # Token IDs not directly supported
                raise HTTPException(
                    status_code=400,
                    detail="Token ID prompts not yet supported"
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
        
        # Generate completions
        all_choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for prompt_idx, prompt in enumerate(prompts):
            request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
            results = await engine.generate(prompt, sampling_params, request_id)
            
            # Format response choices
            for i, output in enumerate(results.outputs):
                all_choices.append(
                    CompletionResponseChoice(
                        text=output.text,
                        index=prompt_idx * request.n + i,
                        finish_reason=_get_finish_reason(output.finish_reason),
                    )
                )
                total_completion_tokens += len(output.token_ids)
            
            total_prompt_tokens += len(results.prompt_token_ids)
        
        return CompletionResponse(
            id=request_id,
            model=request.model,
            choices=all_choices,
            usage={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
        )
        
    except Exception as e:
        logger.error(f"Completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List available models."""
    return ModelListResponse(
        data=[
            ModelObject(
                id=settings.model_name,
                root=settings.model_name,
                owned_by="openai",
            )
        ]
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "liveness": "/livez",
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


# Helper functions
def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string."""
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
            prompt_parts.append(f"{role.capitalize()}: {content}")
    
    # Add final assistant prompt marker
    prompt_parts.append("Assistant:")
    
    return "\n\n".join(prompt_parts)


def _get_finish_reason(vllm_finish_reason: Optional[str]) -> Optional[str]:
    """Convert vLLM finish reason to OpenAI format."""
    if not vllm_finish_reason:
        return None
    
    reason_map = {
        "stop": "stop",
        "length": "length",
        "abort": "stop",
        "error": "stop",
    }
    
    return reason_map.get(vllm_finish_reason, "stop")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
    )