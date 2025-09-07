"""
Core Model Implementation for BWR-DNC 002

This module contains the main Dynamic Neural Core model implementation with clean,
efficient, and well-documented code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Provides stable normalization without learnable parameters, which can
    improve training stability in deep networks.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-8):
        """
        Initialize RMSNorm.
        
        Args:
            d_model: Model dimension
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape [..., d_model]
            
        Returns:
            Normalized tensor of same shape
        """
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embedding.
    
    Implements rotary position embeddings for better long-range modeling
    compared to traditional positional embeddings.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 8192):
        """
        Initialize RoPE embeddings.
        
        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length to support
        """
        super().__init__()
        self.d_model = d_model
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position encodings
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary position embeddings.
        
        Args:
            x: Input tensor of shape [..., seq_len, d_model]
            seq_len: Sequence length (if different from x.shape[-2])
            
        Returns:
            Tensor with rotary embeddings applied
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return self.apply_rope(x, cos, sin)
    
    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor
            cos: Precomputed cosine values
            sin: Precomputed sine values
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # Split into even/odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]
        # Apply rotation
        return torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)


class MultiHeadAttention(nn.Module):
    """
    Efficient Multi-Head Attention with Flash Attention support.
    
    Implements scaled dot-product attention with optional Flash Attention
    for improved memory efficiency on compatible hardware.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            q: Query tensor of shape [batch, q_len, d_model]
            k: Key tensor of shape [batch, k_len, d_model]
            v: Value tensor of shape [batch, v_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch, q_len, d_model]
        """
        B, T_q, _ = q.shape
        _, T_k, _ = k.shape
        
        # Project and reshape
        q = self.q_proj(q).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention with optional Flash Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0+ Flash Attention if available
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # Fallback implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn_output = torch.matmul(attn, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(attn_output)


class DNCBlock(nn.Module):
    """
    Basic DNC Block with Pre-Norm Transformer Architecture.
    
    Implements a transformer block with:
    - Pre-normalization
    - RMSNorm for stability
    - GLU-based feedforward for performance
    """
    
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        """
        Initialize DNC block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            ff_mult: Feedforward expansion multiplier
            dropout: Dropout probability
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        
        # GLU-based feedforward for better performance
        ff_dim = int(ff_mult * d_model)
        self.ff_gate = nn.Linear(d_model, ff_dim, bias=False)
        self.ff_up = nn.Linear(d_model, ff_dim, bias=False)
        self.ff_down = nn.Linear(ff_dim, d_model, bias=False)
        
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DNC block.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        # Pre-norm self-attention
        attn_out = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # GLU feedforward
        gate = F.silu(self.ff_gate(x))  # SwiGLU activation
        up = self.ff_up(x)
        ff_out = self.ff_down(gate * up)
        
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


class DNC(nn.Module):
    """
    Dynamic Neural Core Model.
    
    Implements the main DNC architecture with:
    - Rotary position embeddings
    - Pre-normalized transformer blocks
    - External memory interface
    - Efficient generation
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        n_layers: int = 6, 
        n_heads: int = 8, 
        max_seq_len: int = 8192,
        dropout: float = 0.1
    ):
        """
        Initialize DNC model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token and position embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.rope = RoPEEmbedding(d_model, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DNCBlock(d_model, n_heads, dropout=dropout) 
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.norm_out = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie input/output embeddings
        self.lm_head.weight = self.tok_embed.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Initialize weights with careful scaling.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        input_ids: torch.Tensor,
        external_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the DNC model.
        
        Args:
            input_ids: Input token IDs of shape [batch, seq_len]
            external_memory: Optional external memory to integrate
            
        Returns:
            Tuple of (logits, metadata)
            - logits: Output logits of shape [batch, seq_len, vocab_size]
            - metadata: Dictionary with intermediate representations
        """
        B, T = input_ids.shape
        
        # Token embeddings with position encoding
        x = self.tok_embed(input_ids)
        x = self.rope(x, T)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Integrate external memory if provided
        if external_memory is not None:
            # Simple addition for now - can be made more sophisticated
            x = x + external_memory
        
        # Output prediction
        h = self.norm_out(x)
        logits = self.lm_head(h)
        
        return logits, {"hidden_states": h}
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_length: int = 100, 
        temperature: float = 1.0, 
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate text using the DNC model.
        
        Args:
            input_ids: Input token IDs of shape [batch, seq_len]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold
            
        Returns:
            Generated token IDs of shape [batch, max_length]
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                logits, _ = self(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated