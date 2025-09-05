"""
Debug script for enhanced DNC gradient issues.

This script helps identify where the gradient computation issues are occurring.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

import torch
from enhanced_dnc import EnhancedDNC

def debug_gradients():
    """Debug gradient computation issues."""
    print("Debugging enhanced DNC gradient computation...")
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Create a simple model
    vocab_size = 32
    model = EnhancedDNC(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        num_memory_levels=2,  # Reduce for simpler debugging
        base_memory_slots=32,  # Reduce for simpler debugging
        max_seq_len=16
    )
    
    # Create a very simple dataset
    input_seq = torch.tensor([[0, 1, 2]], dtype=torch.long)
    target_seq = torch.tensor([[1, 2, 31]], dtype=torch.long)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Forward pass
    print("Performing forward pass...")
    logits, _ = model(input_seq)
    print(f"Logits shape: {logits.shape}")
    
    # Compute loss
    print("Computing loss...")
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_seq.view(-1),
        ignore_index=-1
    )
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    print("Performing backward pass...")
    try:
        loss.backward()
        print("Backward pass completed successfully!")
    except Exception as e:
        print(f"Error during backward pass: {e}")
        return False
    
    # Optimizer step
    print("Performing optimizer step...")
    optimizer.step()
    print("Optimizer step completed!")
    
    return True

if __name__ == "__main__":
    debug_gradients()