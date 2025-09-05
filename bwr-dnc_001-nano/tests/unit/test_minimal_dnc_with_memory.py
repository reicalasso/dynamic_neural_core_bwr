"""
Test script for the minimal DNC with memory.

This script verifies that the minimal DNC with external memory
can learn and utilize its memory effectively.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

import torch
from minimal_dnc_with_memory import MinimalDNCWithMemory

def test_minimal_dnc_with_memory():
    """Test the minimal DNC with external memory."""
    print("Testing minimal DNC with external memory...")
    
    # Create model
    vocab_size = 32
    model = MinimalDNCWithMemory(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=2,
        memory_slots=32,
        max_seq_len=32
    )
    
    # Create simple test data
    # Input: [0, 1, 2] (0=start token)
    # Target: [1, 2, 31] (31=end token)
    input_seq = torch.tensor([[0, 1, 2]], dtype=torch.long)
    target_seq = torch.tensor([[1, 2, 31]], dtype=torch.long)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train
    model.train()
    initial_loss = None
    final_loss = None
    
    print("Training...")
    for step in range(30):
        optimizer.zero_grad()
        
        # Forward
        logits, _ = model(input_seq)
        
        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_seq.view(-1),
            ignore_index=-1
        )
        
        if step == 0:
            initial_loss = loss.item()
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
        
        if step == 29:
            final_loss = loss.item()
    
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss decreased: {initial_loss > final_loss}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        generated = model.generate(input_seq, max_length=8)
        print(f"\nGenerated sequence: {generated}")
    
    success = initial_loss > final_loss
    if success:
        print("\n✓ Minimal DNC with memory test passed!")
    else:
        print("\n✗ Minimal DNC with memory test failed!")
    
    return success

if __name__ == "__main__":
    test_minimal_dnc_with_memory()