"""
Test script for the hierarchical memory DNC implementation.

This script tests if the hierarchical memory DNC can learn and utilize
its hierarchical memory structure effectively.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

import torch
from hierarchical_memory_dnc import HierarchicalMemoryDNC

def test_hierarchical_memory_dnc():
    """Test the hierarchical memory DNC implementation."""
    print("Testing hierarchical memory DNC implementation...")
    
    # Create a simple model
    vocab_size = 32
    model = HierarchicalMemoryDNC(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        num_memory_levels=3,
        base_memory_slots=64,
        max_seq_len=16
    )
    
    # Create a very simple dataset - just one example repeated
    # Input: [0, 1, 2, 3] - 0 as start token
    # Target: [1, 2, 3, 31] - 31 as end token
    input_seq = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    target_seq = torch.tensor([[1, 2, 3, 31]], dtype=torch.long)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train for a few steps to see if loss decreases
    model.train()
    initial_loss = None
    final_loss = None
    
    for step in range(50):
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(input_seq)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_seq.view(-1),
            ignore_index=-1  # Don't ignore any tokens
        )
        
        if step == 0:
            initial_loss = loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
        
        if step == 49:
            final_loss = loss.item()
    
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss decreased: {initial_loss > final_loss}")
    
    # Test memory statistics
    memory_stats = model.get_memory_stats()
    print("\nMemory statistics:")
    for level, stats in memory_stats.items():
        print(f"  {level}: {stats}")
    
    if initial_loss > final_loss:
        print("✓ Hierarchical memory DNC model can learn - test passed!")
        return True
    else:
        print("✗ Hierarchical memory DNC model cannot learn - there may be an issue with the implementation")
        return False

if __name__ == "__main__":
    test_hierarchical_memory_dnc()