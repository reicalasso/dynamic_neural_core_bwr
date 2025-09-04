import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class LearnedCompressor(nn.Module):
    """Learned compression module for multi-scale memory."""
    def __init__(self, d_model, compression_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.compression_ratio = compression_ratio
        
        # Perceiver-style cross-attention compressor
        self.latent_dim = d_model // compression_ratio
        self.latents = nn.Parameter(torch.randn(1, self.latent_dim, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Compression/decompression networks
        self.compress_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // compression_ratio),
            nn.LayerNorm(d_model // compression_ratio)
        )
        
        self.decompress_net = nn.Sequential(
            nn.Linear(d_model // compression_ratio, d_model // 2),
            nn.GELU(), 
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x, compress=True):
        """
        x: [B, T, D] input sequence
        compress: if True, compress; if False, decompress
        """
        if compress:
            # Use cross-attention to compress
            B, T, D = x.shape
            latents = self.latents.expand(B, -1, -1)
            compressed, _ = self.cross_attn(latents, x, x)
            return self.compress_net(compressed)
        else:
            return self.decompress_net(x)

class AdvancedStateBank(nn.Module):
    def __init__(self, d_model, slots=2048, heads=8, compression_levels=3):
        super().__init__()
        self.d_model = d_model
        self.slots = slots
        self.heads = heads
        self.compression_levels = compression_levels
        
        # Multi-scale memory banks
        self.levels = nn.ModuleList()
        current_slots = slots
        for level in range(compression_levels):
            level_bank = {
                'K': nn.Parameter(torch.zeros(current_slots, d_model)),
                'V': nn.Parameter(torch.zeros(current_slots, d_model)),
                'salience': nn.Parameter(torch.zeros(current_slots)),
                'age': nn.Parameter(torch.zeros(current_slots)),
                'access_count': nn.Parameter(torch.zeros(current_slots))
            }
            
            # Initialize parameters
            nn.init.xavier_uniform_(level_bank['K'])
            nn.init.xavier_uniform_(level_bank['V'])
            
            self.levels.append(nn.ParameterDict(level_bank))
            current_slots = current_slots // 2  # Hierarchical compression
            
        # Learned compressor for each level
        self.compressors = nn.ModuleList([
            LearnedCompressor(d_model, compression_ratio=2**i) 
            for i in range(compression_levels)
        ])
        
        # Attention routing network
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, compression_levels),
            nn.Softmax(dim=-1)
        )
        
        # Dynamic top-k selection
        self.topk_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def read(self, q, max_topk=64):
        """Advanced hierarchical read with dynamic routing."""
        B, T, D = q.shape
        
        # Predict routing weights and dynamic top-k
        route_weights = self.router(q)  # [B, T, levels]
        dynamic_k = (self.topk_predictor(q) * max_topk).int().clamp(1, max_topk)
        
        all_reads = []
        attention_maps = []
        
        for level_idx, level in enumerate(self.levels):
            level_weight = route_weights[:, :, level_idx:level_idx+1]  # [B, T, 1]
            
            # Compute attention scores
            K = level['K']  # [slots, D]
            V = level['V']
            salience = level['salience']
            
            # Salience-weighted attention
            scores = torch.einsum('btd,sd->bts', q, K) / math.sqrt(D)
            scores = scores + salience.unsqueeze(0).unsqueeze(0)  # Add salience bias
            
            # Dynamic top-k selection per sample
            batch_reads = []
            batch_attns = []
            
            for b in range(B):
                for t in range(T):
                    k = min(dynamic_k[b, t, 0].item(), scores.shape[-1])
                    topk_scores, topk_idx = torch.topk(scores[b:b+1, t:t+1], k=k, dim=-1)
                    
                    # Gather values
                    V_sel = V[topk_idx[0, 0]]  # [k, D]
                    attn = F.softmax(topk_scores[0, 0], dim=-1)  # [k]
                    
                    read = torch.einsum('k,kd->d', attn, V_sel)
                    batch_reads.append(read)
                    batch_attns.append((topk_idx[0, 0], attn))
            
            level_read = torch.stack(batch_reads).view(B, T, D)
            level_read = level_read * level_weight  # Apply routing weight
            all_reads.append(level_read)
            attention_maps.append(batch_attns)
        
        # Combine reads from all levels
        final_read = torch.stack(all_reads, dim=0).sum(dim=0)
        
        return final_read, attention_maps, route_weights

    def write(self, h, attention_maps=None, alpha=0.1):
        """Advanced write with hierarchical updates and learned compression."""
        B, T, D = h.shape
        
        # Route writing to appropriate levels
        route_weights = self.router(h)
        
        for level_idx, level in enumerate(self.levels):
            level_weight = route_weights[:, :, level_idx].mean()
            
            if level_weight > 0.1:  # Only write to significantly weighted levels
                # Compress information for this level
                compressed_h = self.compressors[level_idx](h, compress=True)
                decompressed_h = self.compressors[level_idx](compressed_h, compress=False)
                
                # Simple eviction: replace least salient slots
                pooled = decompressed_h.mean(dim=(0, 1))
                evict_idx = torch.argmin(level['salience'])
                
                with torch.no_grad():
                    level['K'][evict_idx] = pooled.detach()
                    level['V'][evict_idx] = pooled.detach()
                    level['salience'][evict_idx] = (1-alpha) * level['salience'][evict_idx] + alpha
                    level['age'][evict_idx] = 0  # Reset age
                    
                    # Age all other slots
                    mask = torch.ones_like(level['age'], dtype=torch.bool)
                    mask[evict_idx] = False
                    level['age'][mask] += 1

class StateBank(AdvancedStateBank):
    """Backward compatibility wrapper."""
    def __init__(self, d_model, slots=2048, heads=4):
        super().__init__(d_model, slots, heads)

    def read(self, q, topk=32):
        # q: [B, T, D]
        B, T, D = q.shape
        # Using full attention for simplicity, not kNN as described in the plan for this skeleton
        scores = torch.einsum('btd,sd->bts', q, self.K) / (D ** 0.5)
        # scores: [B, T, slots]
        topk_scores, topk_idx = torch.topk(scores, k=min(topk, scores.size(-1)), dim=-1)
        
        # Gather is more efficient than indexing for this case
        # K_sel = self.K[topk_idx] -> Not efficient
        # V_sel = self.V[topk_idx] -> Not efficient
        
        # Instead, we can use the indices to gather from the original V tensor
        # We need to expand topk_idx to match the dimensions of V
        dummy_idx = topk_idx.unsqueeze(-1).expand(B, T, topk, D)
        V_sel = torch.gather(self.V.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1), 2, dummy_idx)

        attn = F.softmax(topk_scores, dim=-1)
        r = torch.einsum('btk,btkd->btd', attn, V_sel)
        return r, topk_idx, attn

    def write(self, h, idx=None, alpha=0.1):
        # h: [B, T, D] → pooled to [D]
        pooled = h.mean(dim=(0,1))
        if idx is None:
            # simple eviction: lowest salience
            evict_idx = torch.argmin(self.salience)
        else:
            # A simple heuristic: replace the least attended slot among the recently read ones
            evict_idx = idx[0, -1, -1] 
            
        with torch.no_grad():
            self.K[evict_idx] = pooled.detach()
            self.V[evict_idx] = pooled.detach()
            # Update salience with EMA
            self.salience[evict_idx] = (1-alpha) * self.salience[evict_idx] + alpha
