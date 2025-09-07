"""
Memory System for BWR-DNC 002

This module implements the external memory system for the DNC, including:
- Hierarchical state banks
- Memory compression
- Efficient read/write operations
- Eviction policies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


@dataclass
class MemorySlot:
    """
    Represents a single memory slot.
    
    Attributes:
        key: Key vector for content-based addressing
        value: Value vector stored in memory
        salience: Importance score (higher = more important)
        age: Time since last access (higher = older)
        access_count: Number of times accessed
    """
    key: torch.Tensor
    value: torch.Tensor
    salience: float = 0.0
    age: int = 0
    access_count: int = 0


class StateBank(nn.Module):
    """
    Hierarchical State Bank for External Memory.
    
    Implements a multi-level memory system with:
    - Hierarchical compression
    - Content-based addressing
    - Salience-based eviction
    - Efficient batch operations
    """
    
    def __init__(
        self, 
        d_model: int, 
        slots_per_level: List[int] = [2048, 1024, 512],
        compression_ratios: List[int] = [1, 2, 4]
    ):
        """
        Initialize hierarchical state bank.
        
        Args:
            d_model: Model dimension
            slots_per_level: Number of slots at each level
            compression_ratios: Compression ratio for each level
        """
        super().__init__()
        self.d_model = d_model
        self.slots_per_level = slots_per_level
        self.compression_ratios = compression_ratios
        self.n_levels = len(slots_per_level)
        
        # Create memory levels
        self.levels = nn.ModuleList()
        for slots, ratio in zip(slots_per_level, compression_ratios):
            level = {
                'keys': nn.Parameter(torch.randn(slots, d_model)),
                'values': nn.Parameter(torch.randn(slots, d_model)),
                'salience': nn.Parameter(torch.zeros(slots)),
                'age': nn.Parameter(torch.zeros(slots, dtype=torch.long)),
                'access_count': nn.Parameter(torch.zeros(slots, dtype=torch.long))
            }
            self.levels.append(nn.ParameterDict(level))
        
        # Initialize parameters
        for level in self.levels:
            nn.init.xavier_uniform_(level['keys'])
            nn.init.xavier_uniform_(level['values'])
    
    def read(self, queries: torch.Tensor, top_k: int = 32) -> torch.Tensor:
        """
        Read from memory using content-based addressing.
        
        Args:
            queries: Query vectors of shape [batch, seq_len, d_model]
            top_k: Number of top matches to retrieve per query
            
        Returns:
            Memory reads of shape [batch, seq_len, d_model]
        """
        B, T, D = queries.shape
        all_reads = []
        
        # Process each level
        for level in self.levels:
            keys = level['keys']  # [slots, d_model]
            values = level['values']  # [slots, d_model]
            salience = level['salience']  # [slots]
            
            # Compute attention scores: [B, T, slots]
            scores = torch.einsum('btd,sd->bts', queries, keys) / math.sqrt(D)
            # Add salience bias
            scores = scores + salience.unsqueeze(0).unsqueeze(0)
            
            # Top-k selection for efficiency
            topk_scores, topk_indices = torch.topk(scores, k=min(top_k, scores.size(-1)), dim=-1)
            
            # Gather values and compute weighted sum
            # Expand indices for gathering values
            gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
            gathered_values = values.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
            selected_values = torch.gather(gathered_values, 2, gather_indices)
            
            # Compute attention weights and weighted sum
            attn_weights = F.softmax(topk_scores, dim=-1)
            reads = torch.einsum('btk,btkd->btd', attn_weights, selected_values)
            all_reads.append(reads)
        
        # Combine reads from all levels (simple sum for now)
        combined_read = torch.stack(all_reads, dim=0).sum(dim=0)
        return combined_read
    
    def write(self, writes: torch.Tensor, salience_updates: Optional[torch.Tensor] = None):
        """
        Write to memory with salience updates.
        
        Args:
            writes: Write vectors of shape [batch, seq_len, d_model]
            salience_updates: Optional salience updates of same shape
        """
        # Pool writes across batch and sequence dimensions
        pooled_writes = writes.mean(dim=(0, 1))  # [d_model]
        
        if salience_updates is not None:
            pooled_salience = salience_updates.mean(dim=(0, 1))  # scalar
        else:
            pooled_salience = torch.tensor(0.1)  # default salience
        
        # Write to each level (simplified approach)
        for level in self.levels:
            # Find least salient slot to evict
            evict_idx = torch.argmin(level['salience'])
            
            with torch.no_grad():
                # Update key and value
                level['keys'][evict_idx] = pooled_writes
                level['values'][evict_idx] = pooled_writes
                
                # Update metadata
                level['salience'][evict_idx] = pooled_salience
                level['age'][evict_idx] = 0
                level['access_count'][evict_idx] += 1
                
                # Age all other slots
                mask = torch.ones_like(level['age'], dtype=torch.bool)
                mask[evict_idx] = False
                level['age'][mask] += 1
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {}
        total_slots = 0
        active_slots = 0
        
        for i, level in enumerate(self.levels):
            slots = level['keys'].shape[0]
            active = (level['salience'] > 0.1).sum().item()
            
            stats[f'level_{i}'] = {
                'slots': slots,
                'active_slots': active,
                'avg_salience': level['salience'].mean().item(),
                'avg_age': level['age'].float().mean().item()
            }
            
            total_slots += slots
            active_slots += active
        
        stats['total_slots'] = total_slots
        stats['active_slots'] = active_slots
        stats['utilization'] = active_slots / total_slots if total_slots > 0 else 0.0
        
        return stats


class CompressedMemory(nn.Module):
    """
    Compressed Memory System with Learned Compression.
    
    Implements memory compression using learned linear transformations
    to reduce memory footprint while maintaining performance.
    """
    
    def __init__(self, d_model: int, compression_ratio: int = 4):
        """
        Initialize compressed memory.
        
        Args:
            d_model: Model dimension
            compression_ratio: Compression ratio (e.g., 4 means 4x compression)
        """
        super().__init__()
        self.d_model = d_model
        self.compression_ratio = compression_ratio
        self.compressed_dim = d_model // compression_ratio
        
        # Compression/decompression networks
        self.compress = nn.Linear(d_model, self.compressed_dim)
        self.decompress = nn.Linear(self.compressed_dim, d_model)
        
        # Initialize with small weights for stable training
        nn.init.normal_(self.compress.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.decompress.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor, compress: bool = True) -> torch.Tensor:
        """
        Apply compression or decompression.
        
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


def create_hierarchical_memory(
    d_model: int, 
    base_slots: int = 2048,
    levels: int = 3
) -> StateBank:
    """
    Create a hierarchical memory system.
    
    Args:
        d_model: Model dimension
        base_slots: Number of slots at the base level
        levels: Number of hierarchical levels
        
    Returns:
        Configured StateBank instance
    """
    slots_per_level = [base_slots // (2 ** i) for i in range(levels)]
    compression_ratios = [2 ** i for i in range(levels)]
    
    return StateBank(
        d_model=d_model,
        slots_per_level=slots_per_level,
        compression_ratios=compression_ratios
    )