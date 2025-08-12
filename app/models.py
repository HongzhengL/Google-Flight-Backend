"""OpenAI-compatible request and response models."""

from typing import Dict, List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field
import time


# Chat Models
class ChatMessage(BaseModel):
    """Chat message format."""
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "gpt-oss-20b"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None


class ChatCompletionResponseChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None
    system_fingerprint: Optional[str] = None


# Streaming Models
class ChatCompletionStreamResponseDelta(BaseModel):
    """Delta content for streaming responses."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    """Streaming chat completion choice."""
    index: int
    delta: ChatCompletionStreamResponseDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]
    usage: Optional[Dict[str, int]] = None
    system_fingerprint: Optional[str] = None


# Completion Models
class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str = "gpt-oss-20b"
    prompt: Union[str, List[str], List[int], List[List[int]]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = Field(default=16, ge=1)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(default=1, ge=1)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None


class CompletionResponseChoice(BaseModel):
    """Completion choice."""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None
    system_fingerprint: Optional[str] = None


# Model List
class ModelObject(BaseModel):
    """Model information object."""
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "openai"
    permission: List[Dict[str, Any]] = []
    root: str
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of available models."""
    object: Literal["list"] = "list"
    data: List[ModelObject]


# Error Models
class ErrorResponse(BaseModel):
    """API error response."""
    error: Dict[str, Any]
    message: str
    type: str
    code: Optional[str] = None