# BWR-DNC Phase 1 Implementation

This is the first phase implementation of the Basic Dynamic Neural Core (DNC) model.

## Overview

This implementation includes:
1. A simplified DNC model with basic memory functionality
2. A simple training pipeline for validation
3. A basic API for serving the model

## Components

- `backend/bwr/basic_dnc.py` - Basic DNC model implementation
- `backend/train_basic_dnc.py` - Training script
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
python train_basic_dnc.py
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

## Next Steps

This is a minimal implementation to validate the core concepts. Future phases will include:
1. Hierarchical memory with multiple levels
2. Advanced memory compression
3. Asynchronous memory updates
4. Curriculum learning training
5. Research visualization dashboard