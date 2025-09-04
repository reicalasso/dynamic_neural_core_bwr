# BWR (BlackWall Reuko) — DNC AI System v2.0

An advanced Dynamic Neural Core implementation with multi-scale memory, learned compression, and real-time visualization dashboard. Optimized for RTX 5060 Mobile and modern GPU architectures.

## 🚀 Key Features

### 🧠 Advanced Dynamic Neural Core
- **Multi-scale memory hierarchy** with 3 compression levels
- **Learned compressor networks** using cross-attention
- **Dynamic routing** between memory levels
- **RoPE positional embeddings** for long-range modeling
- **Mixed precision training** optimized for RTX GPUs

### 📊 Real-time Visualization Dashboard
- **Interactive attention heatmaps** with live updates
- **Memory level distribution** charts and statistics
- **Training metrics** with TensorBoard-style plotting
- **WebSocket live updates** for real-time monitoring
- **State bank explorer** with detailed slot information

### ⚡ Performance Optimizations
- **RTX 5060 specific** configurations and memory management
- **Flash Attention** support for memory efficiency
- **PyTorch 2.0+ compilation** for inference acceleration
- **Gradient accumulation** for effective large batch training
- **Mixed precision** (BFloat16) for stability and speed

### 🌐 Advanced API
- **RESTful endpoints** for model interaction
- **WebSocket support** for real-time updates
- **State management** with persistent memory
- **Checkpoint loading** and model serving
- **CORS enabled** for frontend integration

## 📁 Project Structure

```
BWR-DNC/
├── backend/bwr/           # Core PyTorch DNC implementation
│   ├── model.py          # Enhanced DNC with multi-scale memory
│   ├── statebank.py      # Advanced state management
│   ├── trainer.py        # Optimized training loop
│   └── dataset.py        # Long-range reasoning datasets
├── api/                  # FastAPI server with WebSocket support
│   └── server.py         # Advanced API with live monitoring
├── frontend/app/         # Next.js visualization dashboard
│   ├── components/       # React components for visualization
│   └── pages/           # Dashboard pages
├── configs/             # Model configurations
│   ├── tiny.yaml        # Quick testing (32M params)
│   ├── small.yaml       # Balanced model (120M params)
│   └── rtx5060.yaml     # RTX 5060 optimized (200M params)
└── docker/              # Containerization
```

## 🎯 RTX 5060 Optimizations

This system is specifically optimized for RTX 5060 Mobile (8GB VRAM):

- **Memory-efficient batching** with gradient accumulation
- **BFloat16 mixed precision** for stability
- **Flash Attention** for reduced memory footprint
- **Gradient checkpointing** when needed
- **Optimized DataLoader** with pinned memory
- **PyTorch compilation** for inference speedup

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and navigate
git clone <repository>
cd bwr-dnc

# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# API: http://localhost:8000/docs
```

### Option 2: Local Development

#### Backend Setup

```bash
cd backend
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt

# Quick test with RTX 5060 config
python -m bwr.trainer --config ../configs/rtx5060.yaml --run-name rtx5060_test
```

#### API Server

```bash
cd api
pip install -r requirements.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Dashboard

```bash
cd frontend/app
npm install
npm run dev
```

## 🎮 Usage Examples

### Training with Different Configurations

```bash
# Quick smoke test (32M params)
python -m bwr.trainer --config configs/tiny.yaml --run-name smoke_test

# Balanced training (120M params)
python -m bwr.trainer --config configs/small.yaml --run-name balanced_run

# RTX 5060 optimized (200M params)
python -m bwr.trainer --config configs/rtx5060.yaml --run-name optimized_run
```

### API Usage

```python
import requests

# Generate text
response = requests.post("http://localhost:8000/v1/generate", json={
    "prompt": "Neural networks are",
    "max_length": 50,
    "temperature": 0.8,
    "return_attention": True
})

print(response.json())
```

### Real-time Monitoring

The frontend dashboard provides:

1. **Overview Tab**: Memory statistics and system events
2. **Attention Tab**: Interactive attention heatmaps
3. **Memory Tab**: Multi-level memory visualization
4. **Metrics Tab**: Training progress and performance

## 🔧 Configuration

### Model Sizes

| Config | Parameters | VRAM Usage | Training Speed |
|--------|------------|------------|----------------|
| tiny   | ~32M       | ~2GB       | Fast           |
| small  | ~120M      | ~4GB       | Medium         |
| rtx5060| ~200M      | ~7GB       | Optimized      |

### Memory Management

The system uses hierarchical memory with:
- **Level 0**: Raw memory (1x compression)
- **Level 1**: Compressed memory (2x compression)  
- **Level 2**: Highly compressed (4x compression)

## 📈 Performance Benchmarks

On RTX 5060 Mobile:
- **Training throughput**: ~1200 tokens/sec (mixed precision)
- **Inference speed**: ~2500 tokens/sec (compiled)
- **Memory efficiency**: 75% VRAM utilization
- **Attention computation**: Flash Attention optimized

## 🛠️ Advanced Features

### Curriculum Learning
The training supports multiple long-range reasoning tasks:
- Copy tasks with increasing difficulty
- Key-value lookup in long contexts  
- Pattern completion and infilling
- Needle-in-haystack retrieval
- Associative memory chains

### State Persistence
- Memory banks persist across generations
- State checkpointing and restoration
- Multi-level compression and eviction
- Salience-based memory management

### Real-time Monitoring
- WebSocket live updates
- Attention pattern visualization
- Memory usage tracking
- Training metric dashboards

## 🤝 Contributing

This is an experimental research system. Contributions welcome for:
- Additional long-range reasoning tasks
- Memory management improvements
- Visualization enhancements
- Performance optimizations

## 📄 License

Research and educational use. See LICENSE file for details.

---

**BWR-DNC v2.0** - Advanced Dynamic Neural Core for Long-Range AI 🚀
