"""
BWR-DNC 002: Core Model Integration

This module provides integration between the core DNC model and external memory systems.
It implements clean interfaces for combining different components while maintaining
modularity and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from .model import DNC
from memory.state_bank import StateBank


class MemoryIntegratedDNC(nn.Module):
    """
    DNC Model with Integrated External Memory.
    
    This class combines the core DNC transformer with a hierarchical memory system,
    providing a unified interface for training and inference with external memory.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        memory_slots: List[int] = [2048, 1024, 512],
        memory_integration_layers: List[int] = None
    ):
        """
        Initialize memory-integrated DNC.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            memory_slots: Number of memory slots per level
            memory_integration_layers: Which layers to integrate memory (default: last 2)
        """
        super().__init__()
        
        # Core model
        self.dnc = DNC(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # External memory
        self.memory = StateBank(
            d_model=d_model,
            slots_per_level=memory_slots
        )
        
        # Memory integration layers
        if memory_integration_layers is None:
            memory_integration_layers = [n_layers - 2, n_layers - 1]
        self.memory_integration_layers = set(memory_integration_layers)
        
        # Memory query projection
        self.memory_query_proj = nn.Linear(d_model, d_model)
        self.memory_gate = nn.Linear(d_model * 2, d_model)
        
        # Memory write controller
        self.write_controller = nn.Linear(d_model, d_model)
        self.write_gate = nn.Linear(d_model, 1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        return_memory_stats: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with memory integration.
        
        Args:
            input_ids: Input token IDs of shape [batch, seq_len]
            return_memory_stats: Whether to return memory statistics
            
        Returns:
            Tuple of (logits, metadata)
        """
        B, T = input_ids.shape
        
        # Token embeddings with position encoding
        x = self.dnc.tok_embed(input_ids)
        x = self.dnc.rope(x, T)
        
        memory_reads = []
        
        # Apply transformer blocks with memory integration
        for i, block in enumerate(self.dnc.blocks):
            x = block(x)
            
            # Integrate memory at specified layers
            if i in self.memory_integration_layers:
                # Query memory
                memory_queries = self.memory_query_proj(x)
                memory_output = self.memory.read(memory_queries)
                
                # Gate memory integration
                gate_input = torch.cat([x, memory_output], dim=-1)
                gate = torch.sigmoid(self.memory_gate(gate_input))
                x = x + gate * memory_output
                
                memory_reads.append(memory_output.detach())
        
        # Write to memory using final hidden states
        write_content = self.write_controller(x)
        write_strength = torch.sigmoid(self.write_gate(x))
        
        # Only write if gate is open (> 0.5)
        if write_strength.mean() > 0.5:
            self.memory.write(write_content * write_strength)
        
        # Output prediction
        h = self.dnc.norm_out(x)
        logits = self.dnc.lm_head(h)
        
        # Prepare metadata
        metadata = {
            "hidden_states": h,
            "memory_reads": memory_reads,
            "write_strength": write_strength.mean().item()
        }
        
        if return_memory_stats:
            metadata["memory_stats"] = self.memory.get_memory_stats()
            
        return logits, metadata
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        use_memory: bool = True
    ) -> torch.Tensor:
        """
        Generate text with memory support.
        
        Args:
            input_ids: Input token IDs of shape [batch, seq_len]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold
            use_memory: Whether to use external memory during generation
            
        Returns:
            Generated token IDs of shape [batch, max_length]
        """
        self.eval()
        generated = input_ids.clone()
        
        # Temporarily disable memory integration if requested
        if not use_memory:
            original_layers = self.memory_integration_layers.copy()
            self.memory_integration_layers = set()
        
        try:
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
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)
        
        finally:
            # Restore memory integration settings
            if not use_memory:
                self.memory_integration_layers = original_layers
        
        return generated
    
    def clear_memory(self):
        """Clear all external memory."""
        for level_idx in range(self.memory.n_levels):
            with torch.no_grad():
                getattr(self.memory, f'salience_{level_idx}').zero_()
                getattr(self.memory, f'age_{level_idx}').zero_()
                getattr(self.memory, f'access_count_{level_idx}').zero_()
    
    def get_memory_visualization(self) -> Dict[str, Any]:
        """
        Get data for memory visualization.
        
        Returns:
            Dictionary with visualization data
        """
        viz_data = {}
        
        for level_idx in range(self.memory.n_levels):
            keys = getattr(self.memory, f'keys_{level_idx}').detach().cpu()
            values = getattr(self.memory, f'values_{level_idx}').detach().cpu()
            salience = getattr(self.memory, f'salience_{level_idx}').detach().cpu()
            
            # Compute similarity matrices for visualization
            key_similarity = torch.mm(keys, keys.t())
            value_similarity = torch.mm(values, values.t())
            
            viz_data[f'level_{level_idx}'] = {
                'key_similarity': key_similarity.numpy(),
                'value_similarity': value_similarity.numpy(),
                'salience': salience.numpy(),
                'active_slots': (salience > 0.1).sum().item()
            }
        
        return viz_data


class AdaptiveMemoryController(nn.Module):
    """
    Adaptive Memory Controller.
    
    This module learns when and how to read/write from external memory
    based on the current context and model state.
    """
    
    def __init__(self, d_model: int, memory_dim: int):
        """
        Initialize adaptive memory controller.
        
        Args:
            d_model: Model dimension
            memory_dim: Memory dimension
        """
        super().__init__()
        
        # Read/write decision networks
        self.read_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.write_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Content controllers
        self.read_query_proj = nn.Linear(d_model, memory_dim)
        self.write_content_proj = nn.Linear(d_model, memory_dim)
        
        # Importance scoring
        self.importance_scorer = nn.Linear(d_model, 1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: StateBank
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Control memory operations.
        
        Args:
            hidden_states: Current hidden states [batch, seq_len, d_model]
            memory: External memory system
            
        Returns:
            Tuple of (memory_output, control_signals)
        """
        B, T, D = hidden_states.shape
        
        # Decide whether to read/write
        read_strength = self.read_controller(hidden_states)  # [B, T, 1]
        write_strength = self.write_controller(hidden_states)  # [B, T, 1]
        
        # Generate read queries and write content
        read_queries = self.read_query_proj(hidden_states)  # [B, T, memory_dim]
        write_content = self.write_content_proj(hidden_states)  # [B, T, memory_dim]
        
        # Score importance for selective writing
        importance_scores = self.importance_scorer(hidden_states)  # [B, T, 1]
        
        # Read from memory
        memory_reads = memory.read(read_queries)
        memory_output = read_strength * memory_reads
        
        # Write to memory (only high-importance content)
        write_mask = (write_strength * importance_scores) > 0.5
        if write_mask.any():
            filtered_writes = write_content * write_mask.float()
            memory.write(filtered_writes)
        
        control_signals = {
            'read_strength': read_strength,
            'write_strength': write_strength,
            'importance_scores': importance_scores,
            'write_mask': write_mask.float()
        }
        
        return memory_output, control_signals


def create_integrated_model(
    vocab_size: int,
    model_config: Dict[str, Any] = None,
    memory_config: Dict[str, Any] = None
) -> MemoryIntegratedDNC:
    """
    Factory function to create an integrated DNC model.
    
    Args:
        vocab_size: Vocabulary size
        model_config: Model configuration parameters
        memory_config: Memory configuration parameters
        
    Returns:
        Configured MemoryIntegratedDNC instance
    """
    # Default configurations
    default_model_config = {
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'max_seq_len': 8192,
        'dropout': 0.1
    }
    
    default_memory_config = {
        'memory_slots': [2048, 1024, 512],
        'memory_integration_layers': None
    }
    
    # Merge with user configs
    if model_config:
        default_model_config.update(model_config)
    if memory_config:
        default_memory_config.update(memory_config)
    
    return MemoryIntegratedDNC(
        vocab_size=vocab_size,
        **default_model_config,
        **default_memory_config
    )
