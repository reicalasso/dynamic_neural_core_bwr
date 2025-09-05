"""
Simple API for the basic DNC model.

This provides a minimal FastAPI interface for testing the basic DNC model.
"""

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

from basic_dnc import BasicDNC

app = FastAPI(title="Basic DNC API", version="1.0.0")

# Global model instance
model = None
device = None

class GenerateRequest(BaseModel):
    prompt: List[int]  # List of token IDs
    max_length: int = 50
    temperature: float = 1.0
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    generated_tokens: List[int]
    input_length: int
    output_length: int

@app.on_event("startup")
async def load_model():
    """Load the basic DNC model on startup."""
    global model, device
    
    try:
        print("Loading basic DNC model...")
        
        # Initialize model with simple configuration
        model = BasicDNC(
            vocab_size=128,
            d_model=128,
            n_layers=2,
            n_heads=4,
            slots=64,
            max_seq_len=64
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print("Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Basic DNC API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not set"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text using the basic DNC model."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to tensor
        input_ids = torch.tensor([req.prompt], dtype=torch.long).to(device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=req.max_length,
                temperature=req.temperature,
                top_p=req.top_p
            )
        
        # Convert back to list
        generated_tokens = generated[0].cpu().tolist()
        
        return GenerateResponse(
            generated_tokens=generated_tokens,
            input_length=len(req.prompt),
            output_length=len(generated_tokens) - len(req.prompt)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")