"""
Debug script with forced in-place operations to reproduce the error.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

class TestMemoryBank(torch.nn.Module):
    """A test memory bank that deliberately uses in-place operations."""
    
    def __init__(self, d_model, slots=32):
        super().__init__()
        self.d_model = d_model
        self.slots = slots
        # Make it a parameter so it's included in the optimizer
        self.memory = torch.nn.Parameter(torch.randn(slots, d_model))
        
    def forward(self, x):
        """Forward pass with potential in-place operations."""
        # Deliberately create an in-place operation that could cause issues
        # This simulates what might happen in a real DNC implementation
        B, T, D = x.shape
        
        # Create a view and then try to modify it in-place
        # Note: We can't actually do in-place operations on Parameters directly
        # So we'll create a temporary tensor and perform operations on it
        memory_copy = self.memory.data.clone()  # Clone the data
        memory_view = memory_copy[:min(T, self.slots)]  # This creates a view
        
        # Create a temporary tensor that requires grad
        temp = memory_view.requires_grad_(True)
        
        # In-place operation on the view - this could cause the error
        # But we need to be careful about how we do this
        x_slice = x[0, :temp.shape[0], :]
        result = temp + x_slice  # Non-in-place operation first
        
        # Now do an in-place operation that might cause issues
        # Create a new tensor that shares storage with temp
        result_copy = result.clone()
        result_copy += 0.1  # This modifies result_copy in-place
        
        # Use the modified memory
        output = torch.einsum('btd,sd->bts', x, result_copy)
        return output

def test_forced_inplace_error():
    """Test to force the in-place operation error."""
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Create a simple model with our test memory bank
    d_model = 64
    memory_bank = TestMemoryBank(d_model, slots=32)
    
    # Create optimizer
    optimizer = AdamW(memory_bank.parameters(), lr=1e-3)
    
    # Create sample data that matches the error dimensions
    batch_size, seq_len = 1, 32
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    targets = torch.randint(0, 10, (batch_size, seq_len))
    
    print("Starting forward pass with potential in-place operation...")
    
    try:
        # Forward pass
        output = memory_bank(x)
        
        print("Computing loss...")
        
        # Compute a simple loss
        loss = F.mse_loss(output, torch.randn_like(output))
        
        print(f"Loss computed: {loss.item()}")
        print("Starting backward pass...")
        
        # Backward pass - this should trigger the error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("Backward pass completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forced_inplace_error()