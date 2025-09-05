"""
Simple overfitting test for the basic DNC model.

This script tests if the model can overfit on a very small dataset,
which helps verify that the model and training loop are working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

import torch
from basic_dnc import BasicDNC

def test_overfitting():
    """Test if the model can overfit on a small dataset."""
    print("Testing model overfitting capability...")
    
    # Create a simple model
    vocab_size = 32
    model = BasicDNC(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        slots=64,
        max_seq_len=16
    )
    
    # Create a very simple dataset - just one example repeated
    # Input: [0, 1, 2, 3, 4, 5] - 0 as start token
    # Target: [1, 2, 3, 4, 5, 31] - 31 as end token
    input_seq = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    target_seq = torch.tensor([[1, 2, 3, 4, 5, 31]], dtype=torch.long)
    
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
    
    if initial_loss > final_loss:
        print("✓ Model can learn - overfitting test passed!")
        return True
    else:
        print("✗ Model cannot learn - there may be an issue with the implementation")
        return False

if __name__ == "__main__":
    test_overfitting()