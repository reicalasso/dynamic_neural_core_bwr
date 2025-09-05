"""
Simplified Enhanced DNC Implementation

This implementation focuses on a working version of the enhanced DNC
with a simplified memory mechanism to avoid gradient issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleMemoryBank(nn.Module):
    """Simple memory bank for DNC."""
    
    def __init__(self, d_model, slots=128):
        """
        Args:
            d_model: Model dimension
            slots: Number of memory slots
        """
        super().__init__()
        self.d_model = d_model
        self.slots = slots
        
        # Simple memory matrices
        self.K = nn.Parameter(torch.randn(slots, d_model))  # Key matrix
        self.V = nn.Parameter(torch.randn(slots, d_model))  # Value matrix
        self.salience = nn.Parameter(torch.zeros(slots))    # Importance tracking
        
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
            attention_info: Attention information for analysis
        """
        B, T, D = q.shape
        
        # Compute attention scores between query and keys
        scores = torch.einsum('btd,sd->bts', q, self.K) / math.sqrt(D)
        
        # Apply salience weighting
        scores = scores + self.salience.unsqueeze(0).unsqueeze(0)
        
        # Select top-k slots
        max_slots = min(topk, self.slots)
        topk_scores, topk_indices = torch.topk(scores, k=max_slots, dim=-1)
        
        # Compute attention weights
        attention_weights = F.softmax(topk_scores, dim=-1)
        
        # Gather values for the top-k slots using advanced indexing
        V = self.V  # [slots, D]
        # Create index tensors for advanced indexing
        batch_indices = torch.arange(B, device=q.device).view(B, 1, 1).expand(B, T, max_slots)
        time_indices = torch.arange(T, device=q.device).view(1, T, 1).expand(B, T, max_slots)
        slot_indices = topk_indices  # [B, T, max_slots]
        
        # Expand V to [B, T, slots, D] for proper indexing
        V_expanded = V.unsqueeze(0).unsqueeze(0).expand(B, T, V.shape[0], V.shape[1])
        
        # Use advanced indexing to select the top-k values
        V_selected = V_expanded[batch_indices, time_indices, slot_indices]
        
        # Compute read vectors
        read_vectors = torch.einsum('btk,btkd->btd', attention_weights, V_selected)
        
        return read_vectors, {'attention_weights': attention_weights}
    
    def write(self, h, attention_info=None, alpha=0.1):
        """
        Write to memory by updating key-value pairs.
        
        Args:
            h: Hidden state tensor [B, T, D]
            attention_info: Attention information from read operation
            alpha: Learning rate for salience update
        """
        # Pool hidden states to get a single representation
        pooled = h.mean(dim=(0, 1))  # [D]
        
        # Simple write strategy: update least salient slot
        evict_idx = torch.argmin(self.salience)
        
        with torch.no_grad():
            # Update key and value
            self.K[evict_idx].copy_(pooled.detach())
            self.V[evict_idx].copy_(pooled.detach())
            
            # Update salience with exponential moving average
            new_salience = (1 - alpha) * self.salience[evict_idx] + alpha
            self.salience[evict_idx].copy_(new_salience)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    
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

class EnhancedDNCBlock(nn.Module):
    """Enhanced DNC block."""
    
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

class EnhancedDNC(nn.Module):
    """Enhanced DNC implementation with simplified memory."""
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=8, 
                 slots=128, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            EnhancedDNCBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Memory bank
        self.memory = SimpleMemoryBank(d_model, slots)
        
        # Memory interaction layers
        self.memory_query = nn.Linear(d_model, d_model)
        self.memory_read_proj = nn.Linear(d_model, d_model)
        
        # Output layers
        self.norm_out = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embedding weights
        self.lm_head.weight = self.tok_embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)
    
    def forward(self, input_ids, return_memory_info=False):
        """
        Forward pass through the enhanced DNC model.
        
        Args:
            input_ids: Input token IDs [B, T]
            return_memory_info: Whether to return detailed memory information
            
        Returns:
            logits: Output logits [B, T, vocab_size]
            memory_info: Optional memory information for analysis
        """
        B, T = input_ids.shape
        
        # Token and positional embeddings
        x = self.tok_embed(input_ids)  # [B, T, D]
        pos_emb = self.pos_embed[:T].unsqueeze(0).expand(B, -1, -1)  # [B, T, D]
        x = x + pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Memory interaction
        memory_query = self.memory_query(x)  # [B, T, D]
        memory_read, attention_info = self.memory.read(memory_query)  # [B, T, D]
        memory_read = self.memory_read_proj(memory_read)  # [B, T, D]
        
        # Combine transformer output with memory read
        x = x + memory_read
        
        # Output projection
        x = self.norm_out(x)
        logits = self.lm_head(x)
        
        # Update memory during training
        if self.training:
            self.memory.write(x.detach(), attention_info)
        
        if return_memory_info:
            return logits, attention_info
        
        return logits, {}
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_p=0.9):
        """
        Generate text using the enhanced DNC model.
        
        Args:
            input_ids: Input token IDs [B, T]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            generated_ids: Generated token IDs [B, max_length]
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
                    next_token_logits[indices_to_remove] = -float('inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated

# Example usage
if __name__ == "__main__":
    # Create an enhanced DNC model for testing
    model = EnhancedDNC(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        slots=128,
        max_seq_len=512
    )
    
    # Create sample input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    logits, memory_info = model(input_ids, return_memory_info=True)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Test generation
    generated = model.generate(input_ids[:, :8], max_length=20)
    print(f"Generated sequence shape: {generated.shape}")