"""
DNC blocks for the DNC framework.
"""

import torch.nn as nn
from ..layers.rms_norm import RMSNorm
from ..attention.multihead_attention import MultiHeadAttention

class DNCBlock(nn.Module):
    """A simplified DNC block."""
    
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        
        # Feedforward network
        ff_dim = int(ff_mult * d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        attn_out = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feedforward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x