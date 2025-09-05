"""
Minimal Working DNC with Memory

This is a minimal working implementation of a DNC with external memory
that avoids gradient issues while maintaining core functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MinimalMemoryBank(nn.Module):
    """Minimal external memory bank."""
    
    def __init__(self, d_model, slots=64):
        """
        Args:
            d_model: Model dimension
            slots: Number of memory slots
        """
        super().__init__()
        self.d_model = d_model
        self.slots = slots
        
        # External memory
        self.memory_keys = nn.Parameter(torch.randn(slots, d_model))
        self.memory_values = nn.Parameter(torch.randn(slots, d_model))
        
        # Initialize
        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)
    
    def read(self, query):
        """
        Read from external memory.
        
        Args:
            query: Query tensor [B, T, D]
            
        Returns:
            read_result: Memory read result [B, T, D]
        """
        B, T, D = query.shape
        
        # Compute attention between query and memory keys
        scores = torch.einsum('btd,sd->bts', query, self.memory_keys) / math.sqrt(D)
        
        # Softmax attention
        attn = F.softmax(scores, dim=-1)
        
        # Compute weighted sum of memory values
        read_result = torch.einsum('bts,sd->btd', attn, self.memory_values)
        
        return read_result
    
    def write(self, hidden_state):
        """
        Write to external memory.
        
        Args:
            hidden_state: Hidden state [B, T, D]
        """
        # Pool to get representative vector
        pooled = hidden_state.mean(dim=(0, 1))  # [D]
        
        # Simple strategy: update a random slot
        write_idx = torch.randint(0, self.slots, (1,)).item()
        
        with torch.no_grad():
            # Update memory
            self.memory_keys[write_idx].copy_(pooled)
            self.memory_values[write_idx].copy_(pooled)

class MinimalDNCWithMemory(nn.Module):
    """Minimal DNC with external memory."""
    
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, 
                 memory_slots=64, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # External memory
        self.memory = MinimalMemoryBank(d_model, memory_slots)
        
        # Memory interaction
        self.memory_query = nn.Linear(d_model, d_model)
        self.memory_read_proj = nn.Linear(d_model, d_model)
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Tie weights
        self.lm_head.weight = self.tok_embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
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
            info: Optional memory information
        """
        B, T = input_ids.shape
        
        # Embeddings
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
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Update memory during training
        if self.training:
            self.memory.write(x.detach())
        
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
    model = MinimalDNCWithMemory(
        vocab_size=100,
        d_model=128,
        n_layers=2,
        n_heads=4,
        memory_slots=64,
        max_seq_len=128
    )
    
    # Test
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    logits, _ = model(input_ids, return_memory_info=True)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    generated = model.generate(input_ids[:, :4], max_length=10)
    print(f"Generated shape: {generated.shape}")