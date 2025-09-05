"""
Base memory classes for the DNC framework.
"""

import torch.nn as nn

class BaseMemory(nn.Module):
    """Base class for all memory implementations."""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def read(self, query):
        """
        Read from memory.
        
        Args:
            query: Query tensor [B, T, D]
            
        Returns:
            read_vectors: Read vectors [B, T, D]
            attention_info: Attention information for analysis
        """
        raise NotImplementedError("Subclasses must implement read method")
    
    def write(self, hidden, attention_info=None):
        """
        Write to memory.
        
        Args:
            hidden: Hidden state tensor [B, T, D]
            attention_info: Attention information from read operation
        """
        raise NotImplementedError("Subclasses must implement write method")