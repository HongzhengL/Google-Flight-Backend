"""Configuration management for GPT-OSS-20B API service."""

from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Model configuration
    model_name: str = "gpt-oss-20b"
    huggingface_token: Optional[str] = None
    model_download_path: str = "/tmp/models"
    
    # vLLM engine configuration
    max_model_len: int = 128000
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 4  # Configurable concurrency for user scaling
    tensor_parallel_size: int = 1
    dtype: str = "auto"  # auto, float16, bfloat16, float32
    quantization: Optional[str] = "fp8"  # None, awq, gptq, fp8
    
    # API configuration
    api_title: str = "GPT-OSS-20B API"
    api_version: str = "1.0.0"
    api_description: str = "OpenAI-compatible API for GPT-OSS-20B model"
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Server configuration
    port: int = 8080
    host: str = "0.0.0.0"
    workers: int = 1
    log_level: str = "INFO"
    
    # Cloud Run specific
    k_service: Optional[str] = None  # Set by Cloud Run
    k_revision: Optional[str] = None  # Set by Cloud Run
    cloud_run_timeout: int = 1200  # 20 minutes max for long generations
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()