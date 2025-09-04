import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .statebank import StateBank

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for better stability."""
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding for better long-range modeling."""
    def __init__(self, d_model, max_seq_len=8192):
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
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return self.apply_rope(x, cos, sin)
    
    def apply_rope(self, x, cos, sin):
        # Split into even/odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]
        # Apply rotation
        return torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)

class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with Flash Attention compatibility."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
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
        
        # Attention with optional Flash Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0+ Flash Attention if available
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            # Fallback implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores.masked_fill_(mask == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn_output = torch.matmul(attn, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

class DNCBlock(nn.Module):
    """Enhanced DNC block with better normalization and GLU activation."""
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model)  # RMSNorm for better stability
        
        # GLU-based feedforward for better performance
        ff_dim = int(ff_mult * d_model)
        self.ff_gate = nn.Linear(d_model, ff_dim, bias=False)
        self.ff_up = nn.Linear(d_model, ff_dim, bias=False)
        self.ff_down = nn.Linear(ff_dim, d_model, bias=False)
        
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
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
    """Enhanced Dynamic Neural Core with advanced features."""
    def __init__(self, vocab, d_model=512, n_layers=6, n_heads=8, slots=2048, 
                 max_seq_len=8192, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab
        
        # Token and position embeddings
        self.tok_embed = nn.Embedding(vocab, d_model)
        self.rope = RoPEEmbedding(d_model, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DNCBlock(d_model, n_heads, dropout=dropout) 
            for _ in range(n_layers)
        ])
        
        # State management
        self.state = StateBank(d_model, slots=slots)
        self.state_proj = nn.Linear(d_model, d_model)
        self.read_proj = nn.Linear(d_model, d_model)
        
        # Output heads
        self.norm_out = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        
        # Tie input/output embeddings
        self.lm_head.weight = self.tok_embed.weight
        
        # Training loss components
        self.state_regularizer = nn.MSELoss()
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with careful scaling."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, return_state_info=False):
        B, T = ids.shape
        
        # Token embeddings with position encoding
        x = self.tok_embed(ids)
        x = self.rope(x, T)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # State bank interaction
        q = self.state_proj(x)
        r, attention_maps, route_weights = self.state.read(q)
        r = self.read_proj(r)
        
        # Combine transformer output with state read
        h = x + r
        
        # Output prediction
        h = self.norm_out(h)
        logits = self.lm_head(h)
        
        # Training-time state updates
        if self.training:
            self.state.write(h.detach(), attention_maps)
            
        if return_state_info:
            return logits, {
                "attention_maps": attention_maps,
                "route_weights": route_weights,
                "state_reads": r,
                "hidden_states": h
            }
        
        return logits, {"attention_maps": attention_maps}
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_p=0.9):
        """Optimized generation with state persistence."""
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
                    next_token_logits[indices_to_remove] = -float('inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated
