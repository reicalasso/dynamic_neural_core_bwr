import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class ContextSegment:
    """Represents a segment of context with metadata."""
    content: torch.Tensor
    importance_score: float
    timestamp: float
    segment_id: str
    compression_level: int
    access_count: int

class DynamicCompressionManager:
    """Manages dynamic compression for unlimited context length."""
    
    def __init__(self, base_context_length: int = 512, 
                 max_segments: int = 1000,
                 compression_threshold: float = 0.8):
        self.base_context_length = base_context_length
        self.max_segments = max_segments
        self.compression_threshold = compression_threshold
        
        # Compression levels with different ratios
        self.compression_levels = {
            0: {'ratio': 1.0, 'capacity': base_context_length},
            1: {'ratio': 2.0, 'capacity': base_context_length * 2},
            2: {'ratio': 4.0, 'capacity': base_context_length * 4},
            3: {'ratio': 8.0, 'capacity': base_context_length * 8}
        }
        
        # Active context segments
        self.context_segments: List[ContextSegment] = []
        
        # Compression networks for each level
        self.compression_networks = nn.ModuleDict()
        self.decompression_networks = nn.ModuleDict()
        
        # Statistics
        self.stats = {
            'total_tokens_processed': 0,
            'compression_events': 0,
            'context_hits': 0,
            'context_misses': 0
        }
    
    def initialize_compression_networks(self, d_model: int):
        """Initialize compression/decompression networks."""
        for level in range(4):
            ratio = self.compression_levels[level]['ratio']
            compressed_dim = int(d_model / ratio)
            
            # Compression network
            self.compression_networks[str(level)] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, compressed_dim),
                nn.LayerNorm(compressed_dim)
            )
            
            # Decompression network
            self.decompression_networks[str(level)] = nn.Sequential(
                nn.Linear(compressed_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
    
    def should_compress(self, current_length: int, importance_scores: torch.Tensor) -> bool:
        """Determine if compression should be triggered."""
        # Memory pressure based compression
        memory_pressure = current_length / self.base_context_length
        
        # Importance-based compression
        low_importance_ratio = (importance_scores < 0.3).float().mean()
        
        return (memory_pressure > self.compression_threshold or 
                (memory_pressure > 0.5 and low_importance_ratio > 0.4))
    
    def compute_importance_scores(self, hidden_states: torch.Tensor, 
                                attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute importance scores for each token."""
        B, T, D = hidden_states.shape
        
        # Base importance from hidden state magnitude
        magnitude_scores = torch.norm(hidden_states, dim=-1)  # [B, T]
        magnitude_scores = magnitude_scores / magnitude_scores.max(dim=-1, keepdim=True)[0]
        
        # Attention-based importance
        if attention_weights is not None:
            # Average attention received across all heads
            attention_scores = attention_weights.mean(dim=1).sum(dim=1)  # [B, T]
            attention_scores = attention_scores / attention_scores.max(dim=-1, keepdim=True)[0]
        else:
            attention_scores = torch.ones_like(magnitude_scores)
        
        # Position-based importance (recent tokens are more important)
        position_decay = torch.exp(-0.1 * torch.arange(T, device=hidden_states.device))
        position_scores = position_decay.unsqueeze(0).expand(B, -1)
        
        # Combine scores
        importance = (0.4 * magnitude_scores + 
                     0.4 * attention_scores + 
                     0.2 * position_scores)
        
        return importance
    
    def compress_segment(self, segment: torch.Tensor, target_level: int) -> torch.Tensor:
        """Compress a segment to the target compression level."""
        if target_level == 0:
            return segment  # No compression
        
        compressed = self.compression_networks[str(target_level)](segment)
        return compressed
    
    def decompress_segment(self, compressed_segment: torch.Tensor, 
                          original_level: int) -> torch.Tensor:
        """Decompress a segment from its compression level."""
        if original_level == 0:
            return compressed_segment  # No decompression needed
        
        decompressed = self.decompression_networks[str(original_level)](compressed_segment)
        return decompressed
    
    def adaptive_compression(self, hidden_states: torch.Tensor,
                           attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform adaptive compression of context."""
        B, T, D = hidden_states.shape
        
        # Compute importance scores
        importance = self.compute_importance_scores(hidden_states, attention_weights)
        
        if not self.should_compress(T, importance):
            return hidden_states
        
        # Segment the context
        segments = self._segment_context(hidden_states, importance)
        
        # Compress segments based on importance
        compressed_segments = []
        for segment_data, segment_importance in segments:
            if segment_importance.mean() > 0.7:
                # High importance: no compression
                compressed = segment_data
                compression_level = 0
            elif segment_importance.mean() > 0.4:
                # Medium importance: light compression
                compressed = self.compress_segment(segment_data, 1)
                compression_level = 1
            elif segment_importance.mean() > 0.2:
                # Low importance: medium compression
                compressed = self.compress_segment(segment_data, 2)
                compression_level = 2
            else:
                # Very low importance: high compression
                compressed = self.compress_segment(segment_data, 3)
                compression_level = 3
            
            compressed_segments.append((compressed, compression_level))
        
        # Store segments for later retrieval
        self._store_compressed_segments(compressed_segments)
        
        # Return the most important uncompressed part
        return self._get_working_context(hidden_states, importance)
    
    def _segment_context(self, hidden_states: torch.Tensor, 
                        importance: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Segment context into chunks for compression."""
        B, T, D = hidden_states.shape
        segment_size = self.base_context_length // 4  # 128 tokens per segment
        
        segments = []
        for i in range(0, T, segment_size):
            end_idx = min(i + segment_size, T)
            segment_data = hidden_states[:, i:end_idx]
            segment_importance = importance[:, i:end_idx]
            segments.append((segment_data, segment_importance))
        
        return segments
    
    def _store_compressed_segments(self, compressed_segments: List[Tuple[torch.Tensor, int]]):
        """Store compressed segments for potential future retrieval."""
        import time
        import uuid
        
        for compressed_data, compression_level in compressed_segments:
            segment = ContextSegment(
                content=compressed_data.detach(),
                importance_score=float(torch.rand(1)),  # Would be computed properly
                timestamp=time.time(),
                segment_id=str(uuid.uuid4()),
                compression_level=compression_level,
                access_count=0
            )
            
            self.context_segments.append(segment)
        
        # Maintain segment limit
        if len(self.context_segments) > self.max_segments:
            # Remove oldest, least important segments
            self.context_segments.sort(key=lambda x: x.importance_score * (1.0 / (time.time() - x.timestamp + 1)))
            self.context_segments = self.context_segments[-self.max_segments:]
    
    def _get_working_context(self, hidden_states: torch.Tensor, 
                           importance: torch.Tensor) -> torch.Tensor:
        """Get the working context (most important recent tokens)."""
        B, T, D = hidden_states.shape
        working_size = min(self.base_context_length, T)
        
        # Get most recent tokens as working context
        return hidden_states[:, -working_size:]
    
    def retrieve_relevant_context(self, query: torch.Tensor, 
                                max_retrievals: int = 10) -> Optional[torch.Tensor]:
        """Retrieve relevant compressed context segments."""
        if not self.context_segments:
            return None
        
        # Compute similarity with stored segments
        similarities = []
        for segment in self.context_segments:
            # Simple similarity (would use more sophisticated matching in practice)
            if segment.content.shape[-1] == query.shape[-1]:
                sim = F.cosine_similarity(
                    query.mean(dim=1),  # [B, D]
                    segment.content.mean(dim=1),  # [B, D]
                    dim=-1
                ).mean().item()
            else:
                # Decompress first
                decompressed = self.decompress_segment(segment.content, segment.compression_level)
                sim = F.cosine_similarity(
                    query.mean(dim=1),
                    decompressed.mean(dim=1),
                    dim=-1
                ).mean().item()
            
            similarities.append((sim, segment))
        
        # Sort by similarity and get top segments
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_segments = similarities[:max_retrievals]
        
        # Decompress and return relevant segments
        retrieved_contexts = []
        for sim, segment in top_segments:
            if sim > 0.3:  # Similarity threshold
                if segment.compression_level > 0:
                    decompressed = self.decompress_segment(segment.content, segment.compression_level)
                else:
                    decompressed = segment.content
                
                retrieved_contexts.append(decompressed)
                segment.access_count += 1
                self.stats['context_hits'] += 1
            else:
                self.stats['context_misses'] += 1
        
        if retrieved_contexts:
            return torch.cat(retrieved_contexts, dim=1)
        
        return None


class UnlimitedContextDNC(nn.Module):
    """DNC with unlimited context capability through dynamic compression."""
    
    def __init__(self, base_dnc: nn.Module, base_context_length: int = 512):
        super().__init__()
        self.base_dnc = base_dnc
        self.compression_manager = DynamicCompressionManager(base_context_length)
        
        # Initialize compression networks
        d_model = base_dnc.d_model
        self.compression_manager.initialize_compression_networks(d_model)
        
        # Context assembly layer
        self.context_assembler = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        
        # Learned position embeddings for extended context
        self.extended_pos_embed = nn.Embedding(base_context_length * 8, d_model)
    
    def forward(self, input_ids: torch.Tensor, 
                extended_context: bool = True,
                return_compression_info: bool = False):
        """Forward pass with unlimited context support."""
        B, T = input_ids.shape
        
        # Get embeddings
        x = self.base_dnc.tok_embed(input_ids)
        
        # Apply position embeddings
        if T > self.base_dnc.rope.cos_cached.shape[0] and extended_context:
            # Use extended position embeddings for very long sequences
            pos_ids = torch.arange(T, device=input_ids.device)
            pos_embed = self.extended_pos_embed(pos_ids % self.extended_pos_embed.num_embeddings)
            x = x + pos_embed.unsqueeze(0)
        else:
            # Use standard RoPE
            x = self.base_dnc.rope(x, T)
        
        # Process through transformer blocks
        attention_weights = None
        for block in self.base_dnc.blocks:
            x = block(x)
            # Could extract attention weights here for importance computation
        
        # Dynamic compression if sequence is too long
        if T > self.compression_manager.base_context_length and extended_context:
            x = self.compression_manager.adaptive_compression(x, attention_weights)
            
            # Retrieve relevant historical context
            relevant_context = self.compression_manager.retrieve_relevant_context(x)
            
            if relevant_context is not None:
                # Combine current context with retrieved context
                x, _ = self.context_assembler(x, relevant_context, relevant_context)
        
        # State bank interaction
        q = self.base_dnc.state_proj(x)
        r, attention_maps, route_weights = self.base_dnc.state.read(q)
        r = self.base_dnc.read_proj(r)
        
        # Combine with state read
        h = x + r
        
        # Output
        h = self.base_dnc.norm_out(h)
        logits = self.base_dnc.lm_head(h)
        
        # Training-time state updates
        if self.training:
            self.base_dnc.state.write(h.detach(), attention_maps)
        
        result = {
            "attention_maps": attention_maps,
            "route_weights": route_weights
        }
        
        if return_compression_info:
            result.update({
                "compression_stats": self.compression_manager.stats,
                "active_segments": len(self.compression_manager.context_segments),
                "effective_context_length": x.shape[1]
            })
        
        return logits, result
    
    def get_compression_statistics(self) -> Dict:
        """Get detailed compression statistics."""
        stats = self.compression_manager.stats.copy()
        
        # Add segment analysis
        if self.compression_manager.context_segments:
            levels = [s.compression_level for s in self.compression_manager.context_segments]
            importance_scores = [s.importance_score for s in self.compression_manager.context_segments]
            access_counts = [s.access_count for s in self.compression_manager.context_segments]
            
            stats.update({
                'segments_by_level': {i: levels.count(i) for i in range(4)},
                'avg_importance': sum(importance_scores) / len(importance_scores),
                'avg_access_count': sum(access_counts) / len(access_counts),
                'total_segments': len(self.compression_manager.context_segments)
            })
        
        return stats


def create_unlimited_context_model(base_model: nn.Module, 
                                 context_length: int = 512) -> UnlimitedContextDNC:
    """Create an unlimited context version of a base DNC model."""
    unlimited_model = UnlimitedContextDNC(base_model, context_length)
    return unlimited_model
