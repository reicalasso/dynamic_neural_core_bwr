"""
Simple Hierarchical Memory Implementation

This module implements a simplified hierarchical memory system
that avoids complex gradient issues while maintaining core functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleHierarchicalMemory(nn.Module):
    """Simple hierarchical memory with multiple levels."""
    
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
        
        # Create memory levels with decreasing slots
        self.levels = nn.ModuleList()
        current_slots = base_slots
        
        for level in range(num_levels):
            # Simple parameter dictionary for this level
            level_params = nn.ParameterDict({
                f'keys_{level}': nn.Parameter(torch.randn(current_slots, d_model)),
                f'values_{level}': nn.Parameter(torch.randn(current_slots, d_model)),
                f'salience_{level}': nn.Parameter(torch.zeros(current_slots))
            })
            
            # Initialize parameters
            nn.init.xavier_uniform_(level_params[f'keys_{level}'])
            nn.init.xavier_uniform_(level_params[f'values_{level}'])
            
            self.levels.append(level_params)
            
            # Halve slots for next level (more compression)
            current_slots = max(16, current_slots // 2)
    
    def read(self, query, topk_per_level=4):
        """
        Read from hierarchical memory.
        
        Args:
            query: Query tensor [B, T, D]
            topk_per_level: Top-k slots per level
            
        Returns:
            combined_read: Combined memory read [B, T, D]
            info: Memory information for monitoring
        """
        B, T, D = query.shape
        all_reads = []
        level_weights = []
        
        # Equal weighting for simplicity
        base_weight = 1.0 / self.num_levels
        
        for level_idx, level in enumerate(self.levels):
            # Get parameters for this level
            keys = level[f'keys_{level_idx}']  # [slots, D]
            values = level[f'values_{level_idx}']  # [slots, D]
            salience = level[f'salience_{level_idx}']  # [slots]
            
            # Compute attention between query and keys
            scores = torch.einsum('btd,sd->bts', query, keys) / math.sqrt(D)
            
            # Apply salience weighting
            scores = scores + salience.unsqueeze(0).unsqueeze(0)
            
            # Select top-k slots
            max_slots = min(topk_per_level, keys.shape[0])
            topk_scores, _ = torch.topk(scores, k=max_slots, dim=-1)
            
            # Compute attention weights
            attn_weights = F.softmax(topk_scores, dim=-1)
            
            # Gather values
            # Simple gathering (average of top-k for simplicity)
            gathered_values = values[:max_slots].mean(dim=0)  # [D]
            
            # Expand for batch and time dimensions
            expanded_values = gathered_values.unsqueeze(0).unsqueeze(0).expand(B, T, D)
            
            # Weight by attention
            weighted_read = expanded_values * attn_weights.mean(dim=-1, keepdim=True)
            
            all_reads.append(weighted_read)
            level_weights.append(base_weight)
        
        # Combine reads from all levels
        if all_reads:
            combined_read = torch.stack(all_reads, dim=0).sum(dim=0)
        else:
            combined_read = torch.zeros(B, T, D, device=query.device)
        
        return combined_read, {
            'level_weights': level_weights,
            'num_levels': self.num_levels
        }
    
    def write(self, hidden_state):
        """
        Write to memory (simplified version).
        
        Args:
            hidden_state: Hidden state [B, T, D]
        """
        # Pool to get representative vector
        pooled = hidden_state.mean(dim=(0, 1))  # [D]
        
        # Write to each level (simple update of least salient slots)
        for level_idx, level in enumerate(self.levels):
            # Get parameters for this level
            keys = level[f'keys_{level_idx}']
            values = level[f'values_{level_idx}']
            salience = level[f'salience_{level_idx}']
            
            # Find least salient slot
            least_salient_idx = torch.argmin(salience)
            
            with torch.no_grad():
                # Update key and value
                keys[least_salient_idx].copy_(pooled)
                values[least_salient_idx].copy_(pooled)
                
                # Update salience (simple increment)
                salience[least_salient_idx].add_(0.1)
                
                # Decay other salience values slightly
                mask = torch.ones_like(salience, dtype=torch.bool)
                mask[least_salient_idx] = False
                salience[mask].mul_(0.99)
    
    def get_stats(self):
        """Get memory statistics."""
        stats = {}
        for level_idx, level in enumerate(self.levels):
            # Get parameters for this level
            salience = level[f'salience_{level_idx}']
            
            stats[f'level_{level_idx}'] = {
                'slots': level[f'keys_{level_idx}'].shape[0],
                'avg_salience': salience.mean().item(),
                'active_slots': (salience > 0.1).sum().item()
            }
        return stats

class HierarchicalMemoryDNC(nn.Module):
    """DNC with simple hierarchical memory."""
    
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
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Hierarchical memory
        self.memory = SimpleHierarchicalMemory(d_model, num_memory_levels, base_memory_slots)
        
        # Memory interaction
        self.memory_query = nn.Linear(d_model, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
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
        memory_read, memory_info = self.memory.read(memory_query)  # [B, T, D]
        memory_read = self.memory_proj(memory_read)  # [B, T, D]
        
        # Combine transformer output with memory
        x = x + memory_read
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Update memory during training
        if self.training:
            self.memory.write(x.detach())
        
        if return_memory_info:
            return logits, memory_info
        
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
    
    def get_memory_stats(self):
        """Get memory statistics."""
        return self.memory.get_stats()

# Example usage
if __name__ == "__main__":
    # Create model
    model = HierarchicalMemoryDNC(
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
    
    logits, info = model(input_ids, return_memory_info=True)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Memory levels: {model.num_memory_levels}")
    
    # Memory stats
    stats = model.get_memory_stats()
    print("Memory stats:", stats)