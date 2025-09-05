"""
Direct reproduction of the in-place operation error.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW

def test_direct_inplace_error():
    """Directly reproduce the in-place operation error."""
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Create a tensor that matches the error description: [1, 32, 64]
    x = torch.randn(1, 32, 64, requires_grad=True)
    print(f"Created tensor with shape: {x.shape}")
    
    # Create another tensor for operations
    y = torch.randn(1, 32, 64, requires_grad=True)
    
    # Simulate what might happen in DNC - create a view and modify it in-place
    # This is a common pattern that can cause the error
    x_view = x.view(-1, 64)  # Create a view
    x_slice = x_view[0:10]   # Create another view of the view
    
    # Perform an operation that creates a new tensor
    z = x_slice + y.view(-1, 64)[0:10]
    
    # Now try to do an in-place operation on the view
    # This should trigger the error
    x_slice += z  # This is the in-place operation that causes the issue
    
    # Try to use x in a computation that requires gradients
    loss = x.sum()
    
    print("Starting backward pass...")
    
    try:
        loss.backward()
        print("Backward pass completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

def test_memory_inplace_error():
    """Test in-place operations on memory-like structures."""
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Simulate memory tensor
    memory = torch.randn(32, 64, requires_grad=True)
    
    # Simulate hidden state
    hidden = torch.randn(1, 32, 64, requires_grad=True)
    
    # In the DNC, we might do something like this:
    # Update memory based on hidden state
    
    # This is the problematic pattern:
    # 1. Create a view of memory
    memory_view = memory[0:32]  # View of memory
    
    # 2. Try to update it in-place with hidden state info
    # This is where the error typically occurs
    memory_view += hidden.squeeze(0)  # In-place operation on view
    
    # Try to compute loss
    loss = memory.sum()
    
    print("Starting backward pass...")
    
    try:
        loss.backward()
        print("Backward pass completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Test 1: Direct view in-place operation ===")
    test_direct_inplace_error()
    
    print("\n=== Test 2: Memory in-place operation ===")
    test_memory_inplace_error()