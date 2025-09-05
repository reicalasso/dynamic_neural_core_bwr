"""
Hierarchical memory implementation for Phase 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnedCompressor(nn.Module):
    """Learned compression module for multi-scale memory."""
    
    def __init__(self, d_model, compression_ratio=2):
        super().__init__()
        self.d_model = d_model
        self.compression_ratio = compression_ratio
        
        # Simple linear compression/decompression
        self.compress_net = nn.Sequential(
            nn.Linear(d_model, d_model // compression_ratio),
            nn.LayerNorm(d_model // compression_ratio)
        )
        
        self.decompress_net = nn.Sequential(
            nn.Linear(d_model // compression_ratio, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x, compress=True):
        """
        x: [B, T, D] input sequence
        compress: if True, compress; if False, decompress
        """
        if compress:
            return self.compress_net(x)
        else:
            return self.decompress_net(x)


class HierarchicalMemoryBank(nn.Module):
    """Hierarchical memory bank with multiple levels of compression."""
    
    def __init__(self, d_model, slots=128, levels=3):
        super().__init__()
        self.d_model = d_model
        self.levels = levels
        
        # Multi-scale memory banks (same dimensions for simplicity)
        self.level_banks = nn.ModuleList()
        for level in range(levels):
            # Same number of slots for all levels for simplicity
            bank = nn.ParameterDict({
                'K': nn.Parameter(torch.randn(slots, d_model)),
                'V': nn.Parameter(torch.randn(slots, d_model)),
                'salience': nn.Parameter(torch.zeros(slots))
            })
            
            # Initialize parameters
            nn.init.xavier_uniform_(bank['K'])
            nn.init.xavier_uniform_(bank['V'])
            
            self.level_banks.append(bank)
            
        # Learned compressor for each level (for information compression, not dimension reduction)
        self.compressors = nn.ModuleList([
            LearnedCompressor(d_model, compression_ratio=2**i) 
            for i in range(levels)
        ])
        
        # Attention routing network
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, levels),
            nn.Softmax(dim=-1)
        )
        
    def read(self, q):
        """Advanced hierarchical read with dynamic routing."""
        B, T, D = q.shape
        
        # Predict routing weights
        route_weights = self.router(q)  # [B, T, levels]
        
        all_reads = []
        attention_maps = []
        
        for level_idx, bank in enumerate(self.level_banks):
            level_weight = route_weights[:, :, level_idx:level_idx+1]  # [B, T, 1]
            
            # Compute attention scores (all levels have same dimensions)
            K = bank['K']  # [slots, D]
            V = bank['V']
            salience = bank['salience']
            
            # Salience-weighted attention
            scores = torch.einsum('btd,sd->bts', q, K) / math.sqrt(D)
            scores = scores + salience.unsqueeze(0).unsqueeze(0)  # Add salience bias
            
            # Dynamic top-k selection (sabit k=8)
            k = min(8, scores.shape[-1])
            topk_scores, topk_idx = torch.topk(scores, k=k, dim=-1)
            
            # Compute attention weights
            attention_weights = F.softmax(topk_scores, dim=-1)  # [B, T, k]
            
            # Gather values
            V_selected = V[topk_idx]  # [B, T, k, D]
            read_vectors = torch.einsum('btk,btkd->btd', attention_weights, V_selected)
            
            # Apply routing weight
            weighted_read = read_vectors * level_weight
            all_reads.append(weighted_read)
            attention_maps.append(attention_weights)
        
        # Combine reads from all levels
        final_read = torch.stack(all_reads, dim=0).sum(dim=0)
        
        return final_read, attention_maps
    
    def write(self, h, attention_maps=None, alpha=0.1):
        """Advanced write with hierarchical updates and learned compression."""
        if not self.training:
            return
            
        B, T, D = h.shape
        
        # Route writing to appropriate levels
        route_weights = self.router(h)
        
        for level_idx, (bank, compressor) in enumerate(zip(self.level_banks, self.compressors)):
            level_weight = route_weights[:, :, level_idx].mean()
            
            # Sadece önemli seviyelere yaz
            if level_weight > 0.05:  
                # Bilgiyi sıkıştır
                if level_idx > 0:
                    compressed_h = compressor(h, compress=True)
                    decompressed_h = compressor(compressed_h, compress=False)
                else:
                    decompressed_h = h
                    
                # En az önemli slotu güncelle
                pooled = decompressed_h.mean(dim=(0, 1))  # [D]
                evict_idx = torch.argmin(bank['salience'])
                
                # Avoid in-place operations to prevent gradient computation errors
                K_updated = bank['K'].data.clone()
                V_updated = bank['V'].data.clone()
                salience_updated = bank['salience'].data.clone()
                
                K_updated[evict_idx] = pooled.detach()
                V_updated[evict_idx] = pooled.detach()
                salience_updated[evict_idx] = (1-alpha) * bank['salience'].data[evict_idx] + alpha
                
                # Update the parameters
                with torch.no_grad():
                    bank['K'].data = K_updated
                    bank['V'].data = V_updated
                    bank['salience'].data = salience_updated