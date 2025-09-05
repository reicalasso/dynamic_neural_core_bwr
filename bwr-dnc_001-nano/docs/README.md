# BWR-DNC Phase 1 Implementation

This is the first phase implementation of the Basic Dynamic Neural Core (DNC) model with modular architecture.

## Overview

This implementation includes:
1. A simplified DNC model with basic memory functionality
2. A modular code structure for better maintainability
3. A simple training pipeline for validation
4. A basic API for serving the model

## Modular Components

- `backend/bwr/core/` - Core components (base classes, memory, attention, layers)
- `backend/bwr/models/` - Model implementations (BasicDNC, HierarchicalDNC)
- `backend/bwr/training/` - Training components (datasets, trainers)
- `backend/train_basic_dnc_modular.py` - Modular training script
- `api/main.py` - API server
- `backend/requirements.txt` - Backend dependencies
- `api/requirements.txt` - API dependencies

## Installation

```bash
# Install backend dependencies
pip install -r backend/requirements.txt

# Install API dependencies
pip install -r api/requirements.txt
```

## Usage

### Training

```bash
cd backend
python train_basic_dnc_modular.py
```

### API Server

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Model Architecture

The basic DNC model consists of:
1. Token embeddings
2. Transformer blocks with self-attention
3. Simple memory bank with key-value storage
4. Memory read/write mechanisms
5. Language modeling head

## Modular Structure

The codebase has been restructured into a modular architecture:
- **Core Module**: Base classes and fundamental components
- **Models Module**: Different DNC model implementations
- **Training Module**: Training datasets and trainers
- **Memory Module**: Memory bank implementations
- **Attention Module**: Attention mechanism implementations
- **Layers Module**: Neural network layer implementations

## Next Steps

This implementation validates the core concepts with improved modularity. Future phases will include:
1. Hierarchical memory with multiple levels
2. Advanced memory compression
3. Asynchronous memory updates
4. Curriculum learning training
5. Research visualization dashboard