"""
Attention module for DNC framework.
"""

from .base_attention import BaseAttention
from .multihead_attention import MultiHeadAttention

__all__ = ['BaseAttention', 'MultiHeadAttention']