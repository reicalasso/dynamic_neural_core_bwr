"""
Simple memory implementation for the DNC framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_memory import BaseMemory

class SimpleMemoryBank(BaseMemory):
    """A simplified memory bank for the DNC."""
    
    def __init__(self, d_model, slots=128):
        super().__init__(d_model)
        self.slots = slots
        
        # Memory matrices
        self.K = nn.Parameter(torch.randn(slots, d_model))  # Key matrix
        self.V = nn.Parameter(torch.randn(slots, d_model))  # Value matrix
        
        # Salience tracking
        self.salience = nn.Parameter(torch.zeros(slots))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)
    
    def read(self, q, topk=8):
        """
        Read from memory using attention mechanism.
        
        Args:
            q: Query tensor of shape [B, T, D]
            topk: Number of top slots to attend to
            
        Returns:
            read_vectors: Combined read vectors [B, T, D]
            attention_weights: Attention weights for visualization
        """
        B, T, D = q.shape
        
        # Compute attention scores between query and keys
        scores = torch.einsum('btd,sd->bts', q, self.K) / math.sqrt(D)
        
        # Apply salience weighting
        scores = scores + self.salience.unsqueeze(0).unsqueeze(0)
        
        # Select top-k slots
        topk_scores, topk_indices = torch.topk(scores, k=min(topk, self.slots), dim=-1)
        
        # Compute attention weights
        attention_weights = F.softmax(topk_scores, dim=-1)
        
        # Reshape topk_indices for gathering: [B, T, topk] -> [B*T*topk]
        flat_indices = topk_indices.view(-1)  # [B*T*topk]
        
        # Gather values using the flat indices
        # V has shape [slots, D], we want to select [B*T*topk, D] values
        V_gathered = self.V[flat_indices]  # [B*T*topk, D]
        
        # Reshape back to [B, T, topk, D]
        V_selected = V_gathered.view(B, T, topk, D)
        
        # Compute read vectors
        read_vectors = torch.einsum('btk,btkd->btd', attention_weights, V_selected)
        
        return read_vectors, attention_weights
    
    def write(self, h, attention_weights=None, alpha=0.1):
        """
        Write to memory by updating key-value pairs.
        
        Args:
            h: Hidden state tensor [B, T, D]
            attention_weights: Previous attention weights for write location
            alpha: Learning rate for salience update
        """
        # Pool hidden states to get a single representation
        pooled = h.mean(dim=(0, 1))  # [D]
        
        # Simple write strategy: update least salient slot
        with torch.no_grad():
            evict_idx = torch.argmin(self.salience)
            
            # Update key and value using proper tensor operations
            # Avoid in-place operations to prevent gradient computation errors
            K_updated = self.K.data.clone()
            V_updated = self.V.data.clone()
            salience_updated = self.salience.data.clone()
            
            K_updated[evict_idx] = pooled.detach()
            V_updated[evict_idx] = pooled.detach()
            
            # Update salience with exponential moving average
            if attention_weights is not None:
                # If we have attention weights, use them to update salience
                avg_attention = attention_weights.mean(dim=(0, 1))  # [K]
                if avg_attention.shape[-1] > evict_idx:
                    salience_updated[evict_idx] = (1 - alpha) * self.salience.data[evict_idx] + alpha * avg_attention[evict_idx].detach()
            else:
                # Simple salience update
                salience_updated[evict_idx] = (1 - alpha) * self.salience.data[evict_idx] + alpha
            
            # Update the parameters
            self.K.data = K_updated
            self.V.data = V_updated
            self.salience.data = salience_updated