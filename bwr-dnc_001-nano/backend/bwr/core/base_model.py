"""
Base classes for the DNC framework.
"""

import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all DNC models."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input_ids, return_memory_info=False):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [B, T]
            return_memory_info: Whether to return memory attention weights
            
        Returns:
            logits: Output logits [B, T, vocab_size]
            memory_info: Optional memory information for analysis
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_p=0.9):
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs [B, T]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            generated_ids: Generated token IDs [B, max_length]
        """
        raise NotImplementedError("Subclasses must implement generate method")