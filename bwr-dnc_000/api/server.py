from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sys
import os
import json
import asyncio
from typing import Optional, Dict, List, Callable
import uuid
from datetime import datetime
import logging

# Add backend to path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from bwr.model import DNC
from bwr.research_metrics import ResearchMetricsAggregator, GenerationCounters
from bwr.advanced_research_metrics import AdvancedResearchMetrics
from bwr.advanced_analysis import AdvancedAnalysis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock tokenizer for the skeleton
class MockTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
    def __call__(self, text, return_tensors=None):
        # Simple character-level tokenization for demo
        tokens = [ord(c) % self.vocab_size for c in text[:100]]
        return {'input_ids': torch.tensor(tokens, dtype=torch.long).unsqueeze(0)}
    def decode(self, ids):
        if len(ids.shape) > 1:
            ids = ids.squeeze()
        return ''.join([chr(int(id) % 128 + 32) for id in ids[:50]])

app = FastAPI(title="BWR-DNC Advanced API", version="2.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
state_registry: Dict[str, Dict] = {}
model: Optional[DNC] = None
tokenizer: Optional[MockTokenizer] = None
websocket_connections: List[WebSocket] = []

# Research metrics globals
generation_counters = GenerationCounters()
metrics_aggregator: Optional[ResearchMetricsAggregator] = None
advanced_metrics: Optional[AdvancedResearchMetrics] = None
advanced_analysis: Optional[AdvancedAnalysis] = None
research_ws_clients: List[WebSocket] = []
research_broadcast_task: Optional[asyncio.Task] = None

async def research_broadcast_loop():
    """Background loop broadcasting research metrics periodically."""
    global metrics_aggregator, advanced_metrics, research_ws_clients
    while True:
        try:
            await asyncio.sleep(2)
            if metrics_aggregator and research_ws_clients:
                # Basic metrics
                payload = metrics_aggregator.collect()
                
                # Advanced metrics
                if advanced_metrics:
                    advanced_payload = advanced_metrics.get_comprehensive_metrics()
                    payload["advanced"] = advanced_payload
                
                dead = []
                for ws in research_ws_clients:
                    try:
                        await ws.send_json({"type": "research_metrics", "data": payload})
                    except Exception:
                        dead.append(ws)
                # remove dead connections
                for ws in dead:
                    if ws in research_ws_clients:
                        research_ws_clients.remove(ws)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Research broadcast loop error: {e}")


# Pydantic models
class GenerateRequest(BaseModel):
    prompt: str
    state_id: Optional[str] = None
    max_length: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    return_attention: bool = False

class GenerateResponse(BaseModel):
    text: str
    state_id: str
    state_stats: Dict
    attention_data: Optional[Dict] = None
    generation_info: Dict

class StateCompactRequest(BaseModel):
    state_id: str
    compression_level: int = 1

class ModelStatsResponse(BaseModel):
    total_parameters: int
    memory_usage_mb: float
    model_size_mb: float
    device: str
    dtype: str

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(data))
            except:
                await self.disconnect(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def load_model():
    global model, tokenizer, metrics_aggregator, advanced_metrics, advanced_analysis, research_broadcast_task
    try:
        logger.info("Loading BWR-DNC model...")

        vocab_size = 8000
        model = DNC(
            vocab=vocab_size,
            d_model=512,
            n_layers=6,
            n_heads=8,
            slots=2048,
            max_seq_len=1024
        ).eval()

        tokenizer = MockTokenizer(vocab_size)

        default_state_id = str(uuid.uuid4())
        state_registry[default_state_id] = {
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_access": datetime.now().isoformat(),
            "state_data": model.state.state_dict()
        }

        logger.info("Model loaded successfully!")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        metrics_aggregator = ResearchMetricsAggregator(lambda: model, generation_counters)
        advanced_metrics = AdvancedResearchMetrics(lambda: model, max_history=1000)
        advanced_analysis = AdvancedAnalysis(lambda: model)
        research_broadcast_task = asyncio.create_task(research_broadcast_loop())
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "BWR-DNC Advanced API", "version": "2.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "active_states": len(state_registry),
        "websocket_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/stats", response_model=ModelStatsResponse)
async def get_model_stats():
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate model statistics
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate memory usage (rough approximation)
    param_memory = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
    
    return ModelStatsResponse(
        total_parameters=total_params,
        memory_usage_mb=param_memory,
        model_size_mb=param_memory,
        device=str(next(model.parameters()).device),
        dtype=str(next(model.parameters()).dtype)
    )

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = datetime.now()
        
        # Handle state loading
        if req.state_id and req.state_id in state_registry:
            state_info = state_registry[req.state_id]
            model.state.load_state_dict(state_info["state_data"])
            state_info["access_count"] += 1
            state_info["last_access"] = datetime.now().isoformat()
        
        # Tokenize input
        inputs = tokenizer(req.prompt, return_tensors='pt')['input_ids']
        
        with torch.no_grad():
            if req.return_attention:
                # Generate with attention information
                generated = model.generate(
                    inputs, 
                    max_length=req.max_length,
                    temperature=req.temperature,
                    top_p=req.top_p
                )
                
                # Get attention maps for visualization
                logits, aux_info = model(generated, return_state_info=True)
                attention_data = {
                    "attention_maps": str(aux_info.get("attention_maps", [])),
                    "route_weights": aux_info.get("route_weights", []).tolist() if aux_info.get("route_weights") is not None else [],
                    "memory_reads": aux_info.get("state_reads", []).shape if aux_info.get("state_reads") is not None else []
                }
            else:
                # Fast generation without attention info
                generated = model.generate(
                    inputs,
                    max_length=req.max_length,
                    temperature=req.temperature,
                    top_p=req.top_p
                )
                attention_data = None
        
        # Decode generated text
        generated_text = tokenizer.decode(generated[0])
        
        # Create new state
        new_state_id = str(uuid.uuid4())
        state_registry[new_state_id] = {
            "created_at": datetime.now().isoformat(),
            "access_count": 1,
            "last_access": datetime.now().isoformat(),
            "state_data": model.state.state_dict()
        }
        
        # Generation statistics
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Update generation counters for research metrics
        generation_counters.total_requests += 1
        generation_counters.total_generated_tokens += (generated.shape[1] - inputs.shape[1])
        # Approx recent tokens per second using exponential moving average
        produced = (generated.shape[1] - inputs.shape[1])
        tps = produced / generation_time if generation_time > 0 else 0
        # simple EMA with alpha=0.3
        generation_counters.recent_tokens_per_sec = (
            0.7 * generation_counters.recent_tokens_per_sec + 0.3 * tps
        )

        response = GenerateResponse(
            text=generated_text,
            state_id=new_state_id,
            state_stats={
                "total_slots": model.state.levels[0]['K'].shape[0],
                "active_slots": sum((model.state.levels[0]['salience'] > 0.1).sum().item() for _ in range(len(model.state.levels))),
                "memory_levels": len(model.state.levels),
                "generation_time_sec": generation_time,
                "tokens_generated": generated.shape[1] - inputs.shape[1]
            },
            attention_data=attention_data,
            generation_info={
                "input_length": inputs.shape[1],
                "output_length": generated.shape[1],
                "generation_time": generation_time,
                "tokens_per_second": (generated.shape[1] - inputs.shape[1]) / generation_time if generation_time > 0 else 0
            }
        )
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "generation_complete",
            "data": {
                "prompt": req.prompt[:100],
                "generated_length": generated.shape[1] - inputs.shape[1],
                "time": generation_time,
                "state_id": new_state_id
            }
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/v1/states")
async def list_states():
    """List all active states."""
    return {
        "states": [
            {
                "state_id": sid,
                "created_at": info["created_at"],
                "access_count": info["access_count"],
                "last_access": info["last_access"]
            }
            for sid, info in state_registry.items()
        ],
        "total_states": len(state_registry)
    }

@app.get("/v1/state/{state_id}")
async def get_state_info(state_id: str):
    """Get detailed information about a specific state."""
    if state_id not in state_registry:
        raise HTTPException(status_code=404, detail="State ID not found")
    
    state_info = state_registry[state_id]
    state_data = state_info["state_data"]
    
    # Calculate state statistics
    stats = {}
    for level_idx, level_key in enumerate(['K', 'V', 'salience']):
        if level_key in state_data:
            tensor = state_data[level_key]
            stats[f"level_{level_idx}_{level_key}_shape"] = list(tensor.shape)
            if level_key == 'salience':
                stats[f"level_{level_idx}_active_slots"] = (tensor > 0.1).sum().item()
                stats[f"level_{level_idx}_avg_salience"] = tensor.mean().item()
    
    return {
        "state_id": state_id,
        "metadata": state_info,
        "statistics": stats
    }

@app.post("/v1/state/{state_id}/compact")
async def compact_state(state_id: str, req: StateCompactRequest):
    """Manually trigger state compaction."""
    if state_id not in state_registry:
        raise HTTPException(status_code=404, detail="State ID not found")
    
    # In a real implementation, this would trigger actual compaction
    # For now, simulate compaction
    state_info = state_registry[state_id]
    
    # Broadcast compaction event
    await manager.broadcast({
        "type": "compaction_triggered",
        "data": {
            "state_id": state_id,
            "compression_level": req.compression_level,
            "timestamp": datetime.now().isoformat()
        }
    })
    
    return {
        "message": "Compaction triggered",
        "state_id": state_id,
        "compression_level": req.compression_level
    }

@app.delete("/v1/state/{state_id}")
async def delete_state(state_id: str):
    """Delete a state from memory."""
    if state_id not in state_registry:
        raise HTTPException(status_code=404, detail="State ID not found")
    
    del state_registry[state_id]
    return {"message": "State deleted", "state_id": state_id}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Send periodic state updates
            await asyncio.sleep(2)
            
            if model:
                live_data = {
                    "type": "state_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "active_states": len(state_registry),
                        "model_loaded": True,
                        "memory_stats": {
                            "total_slots": model.state.levels[0]['K'].shape[0] if model.state.levels else 0,
                            "levels": len(model.state.levels) if hasattr(model.state, 'levels') else 1
                        }
                    }
                }
                await websocket.send_text(json.dumps(live_data))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/research/metrics")
async def get_research_metrics():
    if not metrics_aggregator:
        raise HTTPException(status_code=503, detail="Metrics aggregator not initialized")
    return metrics_aggregator.collect()

@app.get("/research/advanced-metrics")
async def get_advanced_research_metrics():
    if not advanced_metrics:
        raise HTTPException(status_code=503, detail="Advanced metrics not initialized")
    return advanced_metrics.get_comprehensive_metrics()

@app.post("/research/training-step")
async def record_training_step(data: dict):
    """Record a training step for metrics collection"""
    if not advanced_metrics:
        raise HTTPException(status_code=503, detail="Advanced metrics not initialized")
    
    loss = data.get("loss", 0.0)
    accuracy = data.get("accuracy", 0.0)
    lr = data.get("learning_rate", 0.0)
    epoch = data.get("epoch", 0)
    
    advanced_metrics.collect_training_step(loss, accuracy, lr, epoch)
    return {"status": "recorded"}

@app.post("/research/analyze-text")
async def analyze_text_inference(data: dict):
    """Analyze text inference for detailed metrics"""
    if not advanced_metrics or not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    text = data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text required")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt')['input_ids']
    
    # Collect detailed inference metrics
    analysis = advanced_metrics.collect_inference_step(inputs)
    
    return {
        "text": text,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/research/gradient-analysis")
async def get_gradient_analysis():
    """Get gradient analysis metrics"""
    if not advanced_metrics:
        raise HTTPException(status_code=503, detail="Advanced metrics not initialized")
    
    return {
        "gradient_norms": advanced_metrics.training_metrics.gradient_norms[-50:],
        "gradient_stability": "stable" if len(advanced_metrics.training_metrics.gradient_norms) == 0 or 
                           all(g < 10.0 for g in advanced_metrics.training_metrics.gradient_norms[-10:]) else "unstable",
        "latest_gradient_norm": advanced_metrics.training_metrics.gradient_norms[-1] if advanced_metrics.training_metrics.gradient_norms else 0.0
    }

@app.get("/research/state-evolution")
async def get_state_evolution():
    """Get state evolution metrics"""
    if not advanced_metrics:
        raise HTTPException(status_code=503, detail="Advanced metrics not initialized")
    
    return {
        "state_changes": advanced_metrics.state_metrics.state_changes[-50:],
        "stability_scores": advanced_metrics.state_metrics.state_stability[-50:],
        "summary": advanced_metrics._get_state_evolution_summary()
    }

@app.get("/research/attention-analysis")
async def get_attention_analysis():
    """Get attention analysis metrics"""
    if not advanced_metrics:
        raise HTTPException(status_code=503, detail="Advanced metrics not initialized")
    
    return {
        "entropy_history": advanced_metrics.attention_metrics.attention_entropy[-50:],
        "concentration_history": advanced_metrics.attention_metrics.attention_concentration[-50:],
        "state_vs_attention_ratios": advanced_metrics.attention_metrics.state_vs_attention_ratio[-50:]
    }

@app.get("/research/efficiency-metrics")
async def get_efficiency_metrics():
    """Get efficiency and performance metrics"""
    if not advanced_metrics:
        raise HTTPException(status_code=503, detail="Advanced metrics not initialized")
    
    return {
        "flops_per_token": advanced_metrics.efficiency_metrics.flops_per_token[-50:],
        "latency_history": advanced_metrics.efficiency_metrics.inference_latency[-50:],
        "memory_scaling": advanced_metrics.efficiency_metrics.memory_per_sequence_length[-20:],
        "summary": advanced_metrics._get_efficiency_summary()
    }

@app.get("/research/state-clustering")
async def get_state_clustering():
    """Get state clustering analysis"""
    if not advanced_analysis:
        raise HTTPException(status_code=503, detail="Advanced analysis not initialized")
    
    try:
        # Use recent state history for clustering
        if not advanced_metrics or not advanced_metrics.state_metrics.state_evolution:
            raise HTTPException(status_code=404, detail="No state history available")
        
        # Get recent states (last 10 steps)
        recent_states = list(advanced_metrics.state_metrics.state_evolution.values())[-10:]
        if not recent_states:
            raise HTTPException(status_code=404, detail="No state data available")
        
        # Convert to tensor format for clustering
        state_tensors = []
        for state_dict in recent_states:
            if 'states' in state_dict:
                # Extract the actual state tensor
                states = state_dict['states']
                if isinstance(states, torch.Tensor):
                    state_tensors.append(states.flatten())
                elif isinstance(states, dict) and 'data' in states:
                    state_tensors.append(torch.tensor(states['data']).flatten())
        
        if not state_tensors:
            raise HTTPException(status_code=404, detail="No valid state tensors found")
        
        # Stack tensors for clustering
        state_matrix = torch.stack(state_tensors)
        
        # Perform clustering analysis
        clusters = advanced_analysis.cluster_states(state_matrix, n_clusters=min(3, len(state_tensors)))
        
        return {
            "clusters": [
                {
                    "id": cluster.cluster_id,
                    "centroid": cluster.centroid.tolist(),
                    "members": cluster.member_indices,
                    "size": cluster.size,
                    "variance": cluster.variance
                }
                for cluster in clusters
            ],
            "total_states": len(state_tensors),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"State clustering error: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@app.post("/research/information-flow")
async def analyze_information_flow(request: dict):
    """Analyze information flow for a given input"""
    if not advanced_analysis:
        raise HTTPException(status_code=503, detail="Advanced analysis not initialized")
    
    try:
        text = request.get('text', '')
        if not text:
            raise HTTPException(status_code=400, detail="Text input required")
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        # Perform information flow analysis
        flow_result = advanced_analysis.analyze_information_flow(input_ids)
        
        return {
            "flow_matrices": [matrix.tolist() for matrix in flow_result.flow_matrices],
            "bottlenecks": [
                {
                    "layer": bottleneck.layer,
                    "position": bottleneck.position,
                    "strength": bottleneck.strength
                }
                for bottleneck in flow_result.bottlenecks
            ],
            "total_flow": flow_result.total_flow,
            "input_text": text,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Information flow analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Information flow analysis failed: {str(e)}")

@app.post("/research/explain-decision")
async def explain_decision(request: dict):
    """Explain model decision for a given input"""
    if not advanced_analysis:
        raise HTTPException(status_code=503, detail="Advanced analysis not initialized")
    
    try:
        text = request.get('text', '')
        target_token_idx = request.get('target_token_idx', -1)  # Last token by default
        
        if not text:
            raise HTTPException(status_code=400, detail="Text input required")
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        # Generate explanation
        explanation = advanced_analysis.explain_decision(input_ids, target_token_idx)
        
        return {
            "importance_scores": explanation.importance_scores.tolist(),
            "state_contributions": explanation.state_contributions.tolist(),
            "attention_contributions": explanation.attention_contributions.tolist(),
            "decision_path": [
                {
                    "layer": step.layer,
                    "operation": step.operation,
                    "importance": step.importance
                }
                for step in explanation.decision_path
            ],
            "confidence": explanation.confidence,
            "input_text": text,
            "target_token_idx": target_token_idx,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Decision explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Decision explanation failed: {str(e)}")

@app.websocket("/ws/research")
async def research_metrics_ws(websocket: WebSocket):
    await websocket.accept()
    research_ws_clients.append(websocket)
    logger.info(f"Research WS connected: {len(research_ws_clients)} active")
    try:
        while True:
            # We don't expect incoming messages, just keep alive pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in research_ws_clients:
            research_ws_clients.remove(websocket)
        logger.info("Research WS disconnected")
    except Exception as e:
        logger.error(f"Research WS error: {e}")
        if websocket in research_ws_clients:
            research_ws_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
