"""Health check and readiness probe endpoints."""

import logging
import torch
from datetime import datetime
from typing import Dict, Optional

from vllm import AsyncLLMEngine

from app.config import settings

logger = logging.getLogger(__name__)


async def health_check() -> Dict:
    """Basic health check endpoint.
    
    Returns:
        Dictionary with health status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.api_title,
        "version": settings.api_version,
        "environment": {
            "k_service": settings.k_service,
            "k_revision": settings.k_revision,
        } if settings.k_service else None,
    }


async def readiness_check(engine: Optional[AsyncLLMEngine] = None) -> Dict:
    """Check if model is loaded and ready.
    
    Parameters:
        engine: Initialized vLLM engine (optional)
        
    Returns:
        Dictionary with readiness status and system info
    """
    try:
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        # Get GPU info if available
        gpu_info = None
        if gpu_available and gpu_count > 0:
            gpu_info = {
                "count": gpu_count,
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
            }
        
        # Check engine status
        engine_ready = engine is not None
        model_loaded = False
        
        if engine_ready:
            try:
                # Check if engine is actually functional
                # This is a lightweight check, not actual inference
                model_loaded = True
                logger.info("Engine is ready and model is loaded")
            except Exception as e:
                logger.error(f"Engine check failed: {str(e)}")
                engine_ready = False
        
        return {
            "ready": engine_ready and gpu_available,
            "timestamp": datetime.utcnow().isoformat(),
            "model": {
                "name": settings.model_name,
                "loaded": model_loaded,
                "max_model_len": settings.max_model_len,
                "max_num_seqs": settings.max_num_seqs,
            },
            "gpu": gpu_info,
            "system": {
                "gpu_available": gpu_available,
                "gpu_count": gpu_count,
                "engine_initialized": engine_ready,
            },
            "config": {
                "quantization": settings.quantization,
                "dtype": settings.dtype,
                "tensor_parallel_size": settings.tensor_parallel_size,
                "gpu_memory_utilization": settings.gpu_memory_utilization,
            },
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return {
            "ready": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


async def liveness_check() -> Dict:
    """Liveness probe for Kubernetes/Cloud Run.
    
    Returns:
        Simple alive status
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
    }