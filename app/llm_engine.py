"""vLLM engine initialization and management."""

import os
import logging
from typing import Optional
from pathlib import Path

from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.engine.arg_utils import AsyncEngineArgs
from huggingface_hub import snapshot_download

from app.config import settings

logger = logging.getLogger(__name__)


class LLMEngineManager:
    """Manages the vLLM engine lifecycle."""
    
    def __init__(self):
        self.engine: Optional[AsyncLLMEngine] = None
        self.model_path: Optional[str] = None
        
    async def initialize_engine(self) -> AsyncLLMEngine:
        """Initialize vLLM engine with GPT-OSS-20B.
        
        Downloads model from Hugging Face if not cached,
        then initializes the vLLM async engine.
        
        Returns:
            Configured AsyncLLMEngine instance
            
        Raises:
            RuntimeError: If model download or initialization fails
        """
        try:
            # Download model if not already cached
            self.model_path = await self._ensure_model_downloaded()
            
            logger.info(f"Initializing vLLM engine with model: {self.model_path}")
            
            # Configure engine arguments
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tokenizer=self.model_path,
                tokenizer_mode="auto",
                trust_remote_code=True,
                dtype=settings.dtype,
                quantization=settings.quantization,
                max_model_len=settings.max_model_len,
                gpu_memory_utilization=settings.gpu_memory_utilization,
                max_num_batched_tokens=settings.max_num_batched_tokens,
                max_num_seqs=settings.max_num_seqs,
                tensor_parallel_size=settings.tensor_parallel_size,
                disable_log_stats=False,
                disable_log_requests=False,
            )
            
            # Create and initialize engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            logger.info("vLLM engine initialized successfully")
            return self.engine
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {str(e)}")
            raise RuntimeError(f"Engine initialization failed: {str(e)}")
    
    async def _ensure_model_downloaded(self) -> str:
        """Ensure model is downloaded and return its path.
        
        Returns:
            Path to the downloaded model
            
        Raises:
            RuntimeError: If model download fails
        """
        # First check if model is pre-downloaded in /models (from Docker build)
        pre_downloaded_path = Path(f"/models/{settings.model_name}")
        if pre_downloaded_path.exists():
            logger.info(f"Using pre-downloaded model from Docker image: {pre_downloaded_path}")
            return str(pre_downloaded_path)
        
        # Fall back to downloading if not pre-downloaded
        model_path = Path(settings.model_download_path) / settings.model_name
        
        # Check if model already exists
        if model_path.exists() and any(model_path.iterdir()):
            logger.info(f"Model already cached at: {model_path}")
            return str(model_path)
        
        logger.info(f"Downloading model {settings.model_name} from Hugging Face...")
        
        try:
            # Create download directory if it doesn't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download model from Hugging Face
            download_kwargs = {
                "repo_id": settings.model_name,
                "local_dir": str(model_path),
                "local_dir_use_symlinks": False,
                "revision": "main",
            }
            
            # Add token if provided
            if settings.huggingface_token:
                download_kwargs["token"] = settings.huggingface_token
            
            snapshot_download(**download_kwargs)
            
            logger.info(f"Model downloaded successfully to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise RuntimeError(f"Model download failed: {str(e)}")
    
    async def shutdown(self):
        """Shutdown the vLLM engine gracefully."""
        if self.engine:
            logger.info("Shutting down vLLM engine...")
            # vLLM engine doesn't have explicit shutdown, will be handled by garbage collection
            self.engine = None
            logger.info("vLLM engine shutdown complete")
    
    def get_engine(self) -> AsyncLLMEngine:
        """Get the current engine instance.
        
        Returns:
            The initialized AsyncLLMEngine
            
        Raises:
            RuntimeError: If engine is not initialized
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize_engine first.")
        return self.engine


# Global engine manager instance
engine_manager = LLMEngineManager()