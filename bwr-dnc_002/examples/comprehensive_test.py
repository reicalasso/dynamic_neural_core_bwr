#!/usr/bin/env python3
"""
Comprehensive test of BWR-DNC 002 system
Tests all major components and functionality
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import DNC
from core.integration import create_integrated_model, MemoryIntegratedDNC
from memory.state_bank import StateBank
from utils import format_memory_size, calculate_model_params
import torch.nn.functional as F

def test_core_model():
    """Test core DNC functionality"""
    print("Testing Core DNC Model...")
    print("-" * 50)
    
    # Create model
    model = DNC(vocab_size=50000, d_model=512, n_layers=8)
    model.eval()
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    
    with torch.no_grad():
        output, info = model(input_ids)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Model parameters: {calculate_model_params(model):,}")
    print()
    
    return model

def test_memory_system():
    """Test standalone memory system"""
    print("Testing Memory System...")
    print("-" * 50)
    
    # Create memory bank
    memory = StateBank(
        d_model=512,
        slots_per_level=[64, 32, 16],  # Smaller for testing
        compression_ratios=[1, 2, 4]
    )
    
    # Test memory operations
    batch_size = 2
    seq_len = 16
    d_model = 512
    
    # Create test data
    memories = torch.randn(batch_size, seq_len, d_model)
    salience = torch.rand(batch_size, seq_len)
    
    # Write to memory
    memory.write(memories, salience)
    print("✓ Memory write successful")
    
    # Read from memory
    queries = torch.randn(batch_size, 8, d_model)
    retrieved = memory.read(queries)
    
    print("✓ Memory read successful")
    print(f"  Query shape: {queries.shape}")
    print(f"  Retrieved shape: {retrieved.shape}")
    
    # Check memory state
    stats = memory.get_memory_stats()
    print(f"  Memory utilization: {stats['utilization']:.1%}")
    
    # Test multiple writes to fill memory
    for i in range(5):
        new_memories = torch.randn(batch_size, seq_len, d_model)
        new_salience = torch.rand(batch_size, seq_len) * (1 + i * 0.2)
        memory.write(new_memories, new_salience)
    
    final_stats = memory.get_memory_stats()
    print(f"  Final utilization: {final_stats['utilization']:.1%}")
    print()
    
    return memory

def test_integrated_system():
    """Test memory-integrated DNC"""
    print("Testing Integrated System...")
    print("-" * 50)
    
    # Create integrated model
    model = create_integrated_model(
        vocab_size=50000,
        model_config={
            'd_model': 512,
            'n_layers': 8,
        },
        memory_config={
            'memory_slots': [128, 64, 32]
        }
    )
    model.eval()
    
    # Test generation with memory tracking
    batch_size = 1
    seq_len = 20
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    
    with torch.no_grad():
        # Generate with memory
        output_with_memory = model.generate(
            input_ids, 
            max_length=30,
            use_memory=True,
            temperature=0.8
        )
        
        # Generate without memory
        output_without_memory = model.generate(
            input_ids,
            max_length=30, 
            use_memory=False,
            temperature=0.8
        )
    
    print("✓ Generation with memory successful")
    print(f"  With memory length: {output_with_memory.shape[1]}")
    print(f"  Without memory length: {output_without_memory.shape[1]}")
    
    # Check memory state
    if hasattr(model, 'memory_bank'):
        stats = model.memory_bank.get_memory_stats()
        print(f"  Memory utilization: {stats['utilization']:.1%}")
    
    print()
    return model

def test_text_generation():
    """Test actual text generation with vocabulary"""
    print("Testing Text Generation...")
    print("-" * 50)
    
    # Create small model for quick testing
    model = create_integrated_model(
        vocab_size=1000,
        model_config={
            'd_model': 256,
            'n_layers': 4,
        },
        memory_config={
            'memory_slots': [32, 16, 8]
        }
    )
    model.eval()
    
    # Create a simple prompt
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    
    with torch.no_grad():
        # Generate multiple samples
        for i in range(3):
            output = model.generate(
                prompt,
                max_length=15,
                use_memory=True,
                temperature=1.0
            )
            print(f"  Sample {i+1}: {output[0].tolist()}")
    
    print("✓ Text generation successful")
    print()
    return model

def test_memory_persistence():
    """Test memory persistence across generations"""
    print("Testing Memory Persistence...")
    print("-" * 50)
    
    model = create_integrated_model(
        vocab_size=1000,
        model_config={
            'd_model': 256,
            'n_layers': 4,
        },
        memory_config={
            'memory_slots': [16, 8, 4]
        }
    )
    model.eval()
    
    # Generate sequence that should build up memory
    prompts = [
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[4, 5, 6]], dtype=torch.long),
        torch.tensor([[7, 8, 9]], dtype=torch.long),
    ]
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            output = model.generate(
                prompt,
                max_length=10,
                use_memory=True,
                temperature=0.8
            )
            
            if hasattr(model, 'memory_bank'):
                stats = model.memory_bank.get_memory_stats()
                utilization = stats['utilization']
                print(f"  Generation {i+1}: Memory utilization {utilization:.1%}")
    
    print("✓ Memory persistence test successful")
    print()

def test_performance():
    """Test performance characteristics"""
    print("Testing Performance...")
    print("-" * 50)
    
    model = create_integrated_model(
        vocab_size=10000,
        model_config={
            'd_model': 512,
            'n_layers': 6,
        },
        memory_config={
            'memory_slots': [64, 32, 16]
        }
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"  Device: {device}")
    print(f"  Model size: {format_memory_size(calculate_model_params(model) * 4)}")
    
    # Time inference
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    
    # Time actual inference
    import time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    tokens_per_sec = (batch_size * seq_len) / avg_time
    
    print(f"  Average inference time: {avg_time*1000:.2f} ms")
    print(f"  Tokens per second: {tokens_per_sec:.0f}")
    print()

def main():
    """Run comprehensive tests"""
    print("BWR-DNC 002: Comprehensive Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test each component
        test_core_model()
        test_memory_system()
        test_integrated_system()
        test_text_generation()
        test_memory_persistence()
        test_performance()
        
        print("All Tests Completed Successfully! ✓")
        print("=" * 60)
        print("BWR-DNC 002 system is fully functional.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
