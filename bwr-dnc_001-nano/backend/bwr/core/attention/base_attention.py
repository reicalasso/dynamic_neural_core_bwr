"""
Base attention classes for the DNC framework.
"""

import torch.nn as nn

class BaseAttention(nn.Module):
    """Base class for all attention implementations."""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, q, k, v, mask=None):
        """
        Compute attention.
        
        Args:
            q: Query tensor [B, T, D]
            k: Key tensor [B, S, D]
            v: Value tensor [B, S, D]
            mask: Optional attention mask
            
        Returns:
            output: Attention output [B, T, D]
        """
        raise NotImplementedError("Subclasses must implement forward method")