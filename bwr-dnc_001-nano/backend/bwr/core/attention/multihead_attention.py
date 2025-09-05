"""
Attention mechanisms for the DNC framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attention import BaseAttention

class MultiHeadAttention(BaseAttention):
    """Simplified multi-head attention."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__(d_model)
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        B, T, C = q.shape
        
        # Project and reshape
        q = self.q_proj(q).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn_output = torch.matmul(attn, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)