"""
BWR-DNC 002: FastAPI Server

This module provides a FastAPI-based REST API for the BWR-DNC model,
offering endpoints for text generation, model management, and research analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import torch
import asyncio
import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.integration import create_integrated_model, MemoryIntegratedDNC
from utils import Config, Logger, get_device, load_checkpoint


# API Models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    input_text: str = Field(..., description="Input text to continue")
    max_length: int = Field(100, ge=1, le=1000, description="Maximum generation length")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p sampling threshold")
    use_memory: bool = Field(True, description="Whether to use external memory")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    input_length: int
    output_length: int
    generation_time: float
    memory_stats: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    device: str
    memory_levels: int
    memory_slots: List[int]


class MemoryStats(BaseModel):
    """Memory statistics response."""
    total_slots: int
    active_slots: int
    utilization: float
    level_stats: Dict[str, Dict[str, Any]]


# Global state
app = FastAPI(
    title="BWR-DNC 002 API",
    description="REST API for the BWR-DNC Dynamic Neural Core model",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[MemoryIntegratedDNC] = None
tokenizer = None  # Would be initialized with actual tokenizer
device = get_device()
logger = Logger("BWR-DNC-API")
config = Config()


async def load_model(model_config: Dict[str, Any] = None):
    """Load the BWR-DNC model asynchronously."""
    global model
    
    try:
        logger.info("Loading BWR-DNC model...")
        
        # Default configuration
        default_config = {
            'vocab_size': 50000,
            'model_config': {
                'd_model': 512,
                'n_layers': 6,
                'n_heads': 8,
                'max_seq_len': 2048,
                'dropout': 0.1
            },
            'memory_config': {
                'memory_slots': [2048, 1024, 512],
                'memory_integration_layers': [4, 5]
            }
        }
        
        if model_config:
            default_config.update(model_config)
        
        model = create_integrated_model(
            vocab_size=default_config['vocab_size'],
            model_config=default_config['model_config'],
            memory_config=default_config['memory_config']
        )
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting BWR-DNC API server...")
    await load_model()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BWR-DNC 002 API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model
    
    is_healthy = model is not None
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": is_healthy,
        "device": str(device)
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    from utils import count_parameters, get_model_size_mb
    
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    return ModelInfo(
        model_name="BWR-DNC 002",
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        model_size_mb=model_size,
        device=str(device),
        memory_levels=len(model.memory.levels),
        memory_slots=[level['keys'].shape[0] for level in model.memory.levels]
    )


@app.get("/model/memory", response_model=MemoryStats)
async def get_memory_stats():
    """Get current memory statistics."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    stats = model.memory.get_memory_stats()
    
    return MemoryStats(
        total_slots=stats['total_slots'],
        active_slots=stats['active_slots'],
        utilization=stats['utilization'],
        level_stats={k: v for k, v in stats.items() if k.startswith('level_')}
    )


@app.post("/model/clear_memory")
async def clear_memory():
    """Clear external memory."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model.clear_memory()
    
    return {"message": "Memory cleared successfully"}


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using the BWR-DNC model."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Convert input text to tokens (simplified - would use actual tokenizer)
        # For now, we'll create dummy tokens
        input_tokens = torch.randint(1, 1000, (1, len(request.input_text.split())))
        input_tokens = input_tokens.to(device)
        
        # Generate
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=input_tokens,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                use_memory=request.use_memory
            )
        
        generation_time = time.time() - start_time
        
        # Convert tokens back to text (simplified)
        generated_text = f"Generated text with {generated_tokens.shape[1]} tokens"
        
        # Get memory stats if requested
        memory_stats = None
        if request.use_memory:
            _, metadata = model(input_tokens, return_memory_stats=True)
            memory_stats = metadata.get('memory_stats')
        
        return GenerationResponse(
            generated_text=generated_text,
            input_length=input_tokens.shape[1],
            output_length=generated_tokens.shape[1],
            generation_time=generation_time,
            memory_stats=memory_stats
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks, model_config: Dict[str, Any] = None):
    """Reload the model with new configuration."""
    global model
    
    try:
        # Clear current model
        model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load new model in background
        background_tasks.add_task(load_model, model_config)
        
        return {"message": "Model reload initiated"}
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@app.get("/research/memory_visualization")
async def get_memory_visualization():
    """Get memory visualization data for research purposes."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        viz_data = model.get_memory_visualization()
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for level_name, data in viz_data.items():
            serializable_data[level_name] = {
                'active_slots': data['active_slots'],
                'salience': data['salience'].tolist(),
                'key_similarity_shape': data['key_similarity'].shape,
                'value_similarity_shape': data['value_similarity'].shape
                # Note: Full similarity matrices might be too large for API response
            }
        
        return serializable_data
        
    except Exception as e:
        logger.error(f"Memory visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@app.get("/research/model_analysis")
async def get_model_analysis():
    """Get detailed model analysis for research purposes."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Run a sample input through the model to get analysis
        sample_input = torch.randint(1, 1000, (1, 32)).to(device)
        
        with torch.no_grad():
            logits, metadata = model(sample_input, return_memory_stats=True)
        
        analysis = {
            'model_info': {
                'output_shape': list(logits.shape),
                'hidden_dim': model.dnc.d_model,
                'num_layers': len(model.dnc.blocks),
                'num_heads': model.dnc.blocks[0].self_attn.n_heads
            },
            'memory_analysis': metadata.get('memory_stats', {}),
            'integration_info': {
                'memory_integration_layers': list(model.memory_integration_layers),
                'write_strength': metadata.get('write_strength', 0.0)
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Model analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BWR-DNC 002 API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, reload=args.reload)
