"""
Memory manager for the DNC framework.
"""

import torch.nn as nn
from .base_memory import BaseMemory
from .simple_memory import SimpleMemoryBank
from .hierarchical_memory import HierarchicalMemoryBank

class MemoryManager:
    """Memory manager for orchestrating different memory types."""
    
    def __init__(self, memory_type="simple", **kwargs):
        self.memory_type = memory_type
        self.memory = self._create_memory(**kwargs)
        self._device = None
    
    def _create_memory(self, **kwargs):
        """Create memory based on type."""
        if self.memory_type == "simple":
            return SimpleMemoryBank(**kwargs)
        elif self.memory_type == "hierarchical":
            return HierarchicalMemoryBank(**kwargs)
        else:
            raise ValueError(f"Unknown memory type: {self.memory_type}")
    
    def to(self, device):
        """Move memory to device."""
        self.memory = self.memory.to(device)
        self._device = device
        return self
    
    def read(self, query):
        """Read from memory."""
        return self.memory.read(query)
    
    def write(self, hidden, attention_info=None):
        """Write to memory."""
        self.memory.write(hidden, attention_info)
    
    def get_memory(self):
        """Get the underlying memory object."""
        return self.memory