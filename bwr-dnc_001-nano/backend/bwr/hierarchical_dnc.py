"""
Hierarchical DNC Implementation for Phase 3

This implementation extends the basic DNC with hierarchical memory functionality
to validate the core concepts of multi-level memory organization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add the current directory to path to import hierarchical_memory
sys.path.append(os.path.dirname(__file__))

from basic_dnc import RMSNorm, SimpleMultiHeadAttention, SimpleDNCBlock
from hierarchical_memory import HierarchicalMemoryBank

class HierarchicalDNC(nn.Module):
    """Hierarchical DNC implementation for Phase 3."""
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, 
                 slots=128, levels=3, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (simple learned positional embeddings)
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleDNCBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Hierarchical memory bank
        self.memory = HierarchicalMemoryBank(d_model, slots, levels)
        
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
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [B, T]
            return_memory_info: Whether to return memory attention weights
            
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
        memory_read, attention_maps = self.memory.read(memory_query)  # [B, T, D]
        memory_read = self.memory_read_proj(memory_read)  # [B, T, D]
        
        # Combine transformer output with memory read
        x = x + memory_read
        
        # Output projection
        x = self.norm_out(x)
        logits = self.lm_head(x)
        
        # Update memory during training
        if self.training:
            self.memory.write(x.detach(), attention_maps)
        
        if return_memory_info:
            return logits, {"attention_maps": attention_maps}
        
        return logits, {}
    
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
    # Create a hierarchical model for testing
    model = HierarchicalDNC(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, slots=64, levels=2)
    
    # Create sample input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    logits, memory_info = model(input_ids, return_memory_info=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of attention maps: {len(memory_info['attention_maps'])}")
    
    # Test generation
    generated = model.generate(input_ids[:, :8], max_length=20)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated sequence: {generated}")