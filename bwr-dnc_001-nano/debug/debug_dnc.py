"""
Debug script to reproduce the in-place operation error in DNC.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

from basic_dnc import BasicDNC

def test_inplace_error():
    """Test to reproduce the in-place operation error."""
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Create model
    model = BasicDNC(vocab_size=64, d_model=256, n_layers=4, n_heads=8, slots=128, max_seq_len=32)
    model.train()
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Create sample data that might trigger the error
    batch_size, seq_len = 32, 16
    input_ids = torch.randint(0, 62, (batch_size, seq_len))
    targets = torch.randint(0, 62, (batch_size, seq_len))
    
    print("Starting forward pass...")
    
    # Forward pass
    logits, _ = model(input_ids, return_memory_info=True)
    
    print("Computing loss...")
    
    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=0
    )
    
    print(f"Loss computed: {loss.item()}")
    print("Starting backward pass...")
    
    # Backward pass - this is where the error should occur
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Backward pass completed successfully!")

if __name__ == "__main__":
    test_inplace_error()