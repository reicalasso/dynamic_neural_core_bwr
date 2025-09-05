"""
Hierarchical Memory Implementation without In-Place Operations

This implementation creates a hierarchical memory system that avoids
in-place operations to prevent gradient computation issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HierarchicalMemory(nn.Module):
    """Hierarchical memory system without in-place operations."""
    
    def __init__(self, d_model, num_levels=3, base_slots=64):
        """
        Args:
            d_model: Model dimension
            num_levels: Number of memory levels (0 = most granular)
            base_slots: Number of slots at the base level
        """
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        
        # Create memory levels with decreasing slots (more compression)
        self.memory_keys = nn.ParameterList()
        self.memory_values = nn.ParameterList()
        
        current_slots = base_slots
        for level in range(num_levels):
            # Memory matrices for this level
            keys = nn.Parameter(torch.randn(current_slots, d_model))
            values = nn.Parameter(torch.randn(current_slots, d_model))
            
            # Initialize parameters
            nn.init.xavier_uniform_(keys)
            nn.init.xavier_uniform_(values)
            
            self.memory_keys.append(keys)
            self.memory_values.append(values)
            
            # Halve slots for next level (more compression)
            current_slots = max(16, current_slots // 2)
    
    def read(self, query):
        """
        Read from hierarchical memory.
        
        Args:
            query: Query tensor [B, T, D]
            
        Returns:
            memory_output: Combined memory read [B, T, D]
        """
        B, T, D = query.shape
        all_level_reads = []
        
        # Equal weighting for all levels
        level_weight = 1.0 / self.num_levels
        
        # Read from each level
        for level_idx in range(self.num_levels):
            keys = self.memory_keys[level_idx]  # [slots, D]
            values = self.memory_values[level_idx]  # [slots, D]
            
            # Compute attention between query and keys
            scores = torch.einsum('btd,sd->bts', query, keys) / math.sqrt(D)
            
            # Softmax attention
            attn = F.softmax(scores, dim=-1)
            
            # Compute weighted sum of values
            read_result = torch.einsum('bts,sd->btd', attn, values)
            
            # Weight by level
            weighted_read = read_result * level_weight
            all_level_reads.append(weighted_read)
        
        # Combine reads from all levels
        if all_level_reads:
            combined_read = torch.stack(all_level_reads, dim=0).sum(dim=0)
        else:
            combined_read = torch.zeros_like(query)
        
        return combined_read

class HierarchicalDNC(nn.Module):
    """DNC with hierarchical memory."""
    
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4,
                 num_memory_levels=3, base_memory_slots=64, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_memory_levels = num_memory_levels
        
        # Token embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer layers using built-in PyTorch layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Hierarchical memory
        self.memory = HierarchicalMemory(d_model, num_memory_levels, base_memory_slots)
        
        # Memory interaction
        self.memory_query = nn.Linear(d_model, d_model)
        self.memory_read_proj = nn.Linear(d_model, d_model)
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Tie embedding weights
        self.lm_head.weight = self.tok_embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, return_memory_info=False):
        """
        Forward pass.
        
        Args:
            input_ids: Input tokens [B, T]
            return_memory_info: Return memory statistics
            
        Returns:
            logits: Output logits [B, T, vocab_size]
            memory_info: Optional memory information
        """
        B, T = input_ids.shape
        
        # Token and positional embeddings
        x = self.tok_embed(input_ids)  # [B, T, D]
        pos_emb = self.pos_embed[:T].unsqueeze(0).expand(B, -1, -1)  # [B, T, D]
        x = x + pos_emb
        
        # Transformer processing
        x = self.transformer(x)  # [B, T, D]
        
        # Memory interaction
        memory_query = self.memory_query(x)  # [B, T, D]
        memory_read = self.memory.read(memory_query)  # [B, T, D]
        memory_read = self.memory_read_proj(memory_read)  # [B, T, D]
        
        # Combine transformer output with memory
        x = x + memory_read
        
        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        if return_memory_info:
            return logits, {}
        
        return logits, {}
    
    def generate(self, input_ids, max_length=50, temperature=1.0):
        """Generate text."""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                logits, _ = self(generated)
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

# Example usage
if __name__ == "__main__":
    # Create model
    model = HierarchicalDNC(
        vocab_size=100,
        d_model=128,
        n_layers=2,
        n_heads=4,
        num_memory_levels=3,
        base_memory_slots=64,
        max_seq_len=128
    )
    
    # Test
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    logits, _ = model(input_ids, return_memory_info=True)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Memory levels: {model.num_memory_levels}")
    
    # Test generation
    generated = model.generate(input_ids[:, :4], max_length=10)
    print(f"Generated shape: {generated.shape}")