"""
Training configuration for DNC models.
"""

def create_basic_config():
    """Create configuration for basic DNC training."""
    return {
        'model': {
            'vocab_size': 64,   # Smaller vocabulary for easier learning
            'd_model': 256,     # Model dimension
            'n_layers': 4,      # Number of transformer layers
            'n_heads': 8,       # Number of attention heads
            'slots': 128,       # Memory slots
            'max_seq_len': 32   # Maximum sequence length
        },
        'training': {
            'batch_size': 32,   # Batch size
            'lr': 1e-3,        # Learning rate
            'weight_decay': 0.01,
            'epochs': 20,       # Number of epochs
            'grad_clip': 1.0
        }
    }


def create_hierarchical_config():
    """Create configuration for hierarchical DNC training."""
    return {
        'model': {
            'vocab_size': 64,   # Smaller vocabulary for easier learning
            'd_model': 256,     # Model dimension
            'n_layers': 4,      # Number of transformer layers
            'n_heads': 8,       # Number of attention heads
            'slots': 128,       # Memory slots
            'levels': 3,        # Hierarchical memory levels
            'max_seq_len': 32   # Maximum sequence length
        },
        'training': {
            'batch_size': 32,   # Batch size
            'lr': 1e-3,        # Learning rate
            'weight_decay': 0.01,
            'epochs': 15,       # Number of epochs
            'grad_clip': 1.0
        }
    }