"""
Hierarchical Memory Bank Implementation

This module implements a hierarchical memory bank with multiple levels
of compression for the enhanced DNC model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Compressor(nn.Module):
    """Learned compressor for hierarchical memory."""
    
    def __init__(self, input_dim, compression_ratio=2):
        """
        Args:
            input_dim: Input dimension
            compression_ratio: Compression ratio (e.g., 2 means half the size)
        """
        super().__init__()
        self.input_dim = input_dim
        self.compression_ratio = compression_ratio
        self.compressed_dim = max(32, input_dim // compression_ratio)
        
        # Simple linear compression
        self.compress = nn.Linear(input_dim, self.compressed_dim)
        self.decompress = nn.Linear(self.compressed_dim, input_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.compress.weight)
        nn.init.xavier_uniform_(self.decompress.weight)
    
    def forward(self, x, compress=True):
        """
        Compress or decompress data.
        
        Args:
            x: Input tensor
            compress: If True, compress; if False, decompress
            
        Returns:
            Compressed or decompressed tensor
        """
        if compress:
            return self.compress(x)
        else:
            return self.decompress(x)

class HierarchicalMemoryBank(nn.Module):
    """Hierarchical memory bank with multiple compression levels."""
    
    def __init__(self, d_model, num_levels=3, base_slots=128):
        """
        Args:
            d_model: Model dimension
            num_levels: Number of memory levels (0 = most granular, higher = more compressed)
            base_slots: Number of slots at the base level
        """
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        
        # Create memory levels with increasing compression
        self.levels = nn.ModuleList()
        self.compressors = nn.ModuleList()
        
        current_slots = base_slots
        for level in range(num_levels):
            # Memory matrices for this level
            level_bank = nn.ParameterDict({
                'K': nn.Parameter(torch.randn(current_slots, d_model)),  # Key matrix
                'V': nn.Parameter(torch.randn(current_slots, d_model)),  # Value matrix
                'salience': nn.Parameter(torch.zeros(current_slots)),     # Importance tracking
                'age': nn.Parameter(torch.zeros(current_slots)),          # Age tracking
                'access_count': nn.Parameter(torch.zeros(current_slots))  # Access frequency
            })
            
            # Initialize parameters
            nn.init.xavier_uniform_(level_bank['K'])
            nn.init.xavier_uniform_(level_bank['V'])
            
            self.levels.append(level_bank)
            
            # Add compressor for this level (except base level)
            if level > 0:
                compression_ratio = 2 ** level
                compressor = Compressor(d_model, compression_ratio)
                self.compressors.append(compressor)
            
            # Halve the slots for next level (more compression)
            current_slots = max(32, current_slots // 2)
    
    def read(self, q, topk_per_level=4):
        """
        Read from hierarchical memory using attention mechanism.
        
        Args:
            q: Query tensor of shape [B, T, D]
            topk_per_level: Number of top slots to attend to per level
            
        Returns:
            read_vectors: Combined read vectors [B, T, D]
            attention_info: Detailed attention information for analysis
        """
        B, T, D = q.shape
        all_reads = []
        attention_weights_list = []
        level_contributions = []
        
        # Compute level routing weights (equal weighting for simplicity)
        level_weights = torch.ones(self.num_levels, device=q.device) / self.num_levels
        
        for level_idx, level in enumerate(self.levels):
            # Compute attention scores between query and keys
            K = level['K']  # [slots, D]
            scores = torch.einsum('btd,sd->bts', q, K) / math.sqrt(D)
            
            # Apply salience weighting
            scores = scores + level['salience'].unsqueeze(0).unsqueeze(0)
            
            # Select top-k slots for this level
            max_slots = min(topk_per_level, K.shape[0])
            topk_scores, topk_indices = torch.topk(scores, k=max_slots, dim=-1)
            
            # Compute attention weights for this level
            level_attention = F.softmax(topk_scores, dim=-1)
            attention_weights_list.append(level_attention)
            
            # Gather values for the top-k slots
            V = level['V']  # [slots, D]
            # Use advanced indexing to gather values
            batch_indices = torch.arange(B, device=q.device).view(B, 1, 1).expand(B, T, max_slots)
            time_indices = torch.arange(T, device=q.device).view(1, T, 1).expand(B, T, max_slots)
            slot_indices = topk_indices  # [B, T, max_slots]
            
            # Expand V for proper indexing
            V_expanded = V.unsqueeze(0).unsqueeze(0).expand(B, T, V.shape[0], V.shape[1])
            
            # Gather the top-k values
            V_selected = V_expanded[batch_indices, time_indices, slot_indices]
            
            # Compute read vectors for this level
            level_read = torch.einsum('btk,btkd->btd', level_attention, V_selected)
            
            # Apply level weight
            weighted_read = level_read * level_weights[level_idx]
            all_reads.append(weighted_read)
            level_contributions.append(level_weights[level_idx].item())
        
        # Combine reads from all levels
        if all_reads:
            final_read = torch.stack(all_reads, dim=0).sum(dim=0)
        else:
            final_read = torch.zeros_like(q)
        
        return final_read, {
            'attention_weights': attention_weights_list,
            'level_contributions': level_contributions
        }
    
    def write(self, h, attention_info=None, alpha=0.1):
        """
        Write to hierarchical memory by updating key-value pairs.
        
        Args:
            h: Hidden state tensor [B, T, D]
            attention_info: Attention information from read operation
            alpha: Learning rate for salience update
        """
        # Pool hidden states to get a single representation
        pooled = h.mean(dim=(0, 1))  # [D]
        
        # Write to each level
        for level_idx, level in enumerate(self.levels):
            # Simple write strategy: update least salient slot
            evict_idx = torch.argmin(level['salience'])
            
            # Compress data for this level (except base level)
            if level_idx == 0:
                # Base level - no compression
                data_to_store = pooled
            else:
                # Higher levels - compress data
                compressor_idx = level_idx - 1
                if compressor_idx < len(self.compressors):
                    data_to_store = self.compressors[compressor_idx](pooled, compress=True)
                    # Decompress for storage consistency
                    data_to_store = self.compressors[compressor_idx](data_to_store, compress=False)
                else:
                    data_to_store = pooled
            
            with torch.no_grad():
                # Update key and value
                level['K'][evict_idx].copy_(data_to_store.detach())
                level['V'][evict_idx].copy_(data_to_store.detach())
                
                # Update salience with exponential moving average
                new_salience = (1 - alpha) * level['salience'][evict_idx] + alpha
                level['salience'][evict_idx].copy_(new_salience)
                
                # Update age and access count
                level['age'][:].add_(1)  # Increment all ages
                level['age'][evict_idx].copy_(torch.tensor(0.0))  # Reset age of written slot
                level['access_count'][evict_idx].add_(1)  # Increment access count
    
    def get_memory_stats(self):
        """Get memory statistics for monitoring."""
        stats = {}
        for level_idx, level in enumerate(self.levels):
            stats[f'level_{level_idx}'] = {
                'slots': level['K'].shape[0],
                'active_slots': (level['salience'] > 0.1).sum().item(),
                'avg_salience': level['salience'].mean().item(),
                'avg_age': level['age'].mean().item(),
                'total_accesses': level['access_count'].sum().item()
            }
        return stats

class HierarchicalMemoryDNC(nn.Module):
    """DNC with hierarchical memory bank."""
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=8, 
                 num_memory_levels=3, base_memory_slots=128, max_seq_len=512):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            num_memory_levels: Number of hierarchical memory levels
            base_memory_slots: Number of slots at base memory level
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_memory_levels = num_memory_levels
        
        # Token embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            self._create_transformer_block(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Hierarchical memory bank
        self.memory = HierarchicalMemoryBank(d_model, num_memory_levels, base_memory_slots)
        
        # Memory interaction layers
        self.memory_query = nn.Linear(d_model, d_model)
        self.memory_read_proj = nn.Linear(d_model, d_model)
        
        # Output layers
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embedding weights
        self.lm_head.weight = self.tok_embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_transformer_block(self, d_model, n_heads):
        """Create a transformer block."""
        class TransformerBlock(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                self.norm1 = nn.LayerNorm(d_model)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                # Self-attention
                attn_out, _ = self.attn(x, x, x)
                x = x + self.dropout(attn_out)
                x = self.norm1(x)
                
                # Feedforward
                ff_out = self.ff(x)
                x = x + self.dropout(ff_out)
                x = self.norm2(x)
                return x
        
        return TransformerBlock(d_model, n_heads)
    
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
        Forward pass through the DNC with hierarchical memory.
        
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
        Generate text using the DNC with hierarchical memory.
        
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
    
    def get_memory_stats(self):
        """Get hierarchical memory statistics."""
        return self.memory.get_memory_stats()

# Example usage
if __name__ == "__main__":
    # Create a hierarchical memory DNC model for testing
    model = HierarchicalMemoryDNC(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        num_memory_levels=3,
        base_memory_slots=128,
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
    print(f"Number of memory levels: {model.num_memory_levels}")
    
    # Test generation
    generated = model.generate(input_ids[:, :8], max_length=20)
    print(f"Generated sequence shape: {generated.shape}")
    
    # Test memory statistics
    memory_stats = model.get_memory_stats()
    print("Memory statistics:")
    for level, stats in memory_stats.items():
        print(f"  {level}: {stats}")