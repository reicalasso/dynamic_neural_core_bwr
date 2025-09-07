"""
BWR-DNC 002 Core Module

Contains the core model implementations and integration layers.
"""

from .model import DNC, DNCBlock, MultiHeadAttention, RMSNorm, RoPEEmbedding
from .integration import MemoryIntegratedDNC, create_integrated_model, AdaptiveMemoryController

__all__ = [
    'DNC',
    'DNCBlock', 
    'MultiHeadAttention',
    'RMSNorm',
    'RoPEEmbedding',
    'MemoryIntegratedDNC',
    'create_integrated_model',
    'AdaptiveMemoryController'
]
