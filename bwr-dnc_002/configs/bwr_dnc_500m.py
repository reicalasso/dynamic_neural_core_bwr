"""
BWR-DNC 500M Model Configuration

Configuration for creating a 500 million parameter BWR-DNC model.
This configuration is optimized for mid-range GPUs (RTX 3080/4070, etc.)
"""

# 500M Parameter Configuration
MODEL_500M_CONFIG = {
    'vocab_size': 50000,
    'd_model': 1024,        # Model dimension
    'n_layers': 31,         # Fine-tuned for ~500M params
    'n_heads': 16,          # 16 attention heads
    'max_seq_len': 8192,
    'dropout': 0.1
}

# Memory configuration for 500M model
MEMORY_500M_CONFIG = {
    'memory_slots': [4096, 2048, 1024],  # Smaller memory
    'memory_integration_layers': [16, 17, 18, 19]  # Last 4 layers
}

# Training configuration for 500M model
TRAINING_500M_CONFIG = {
    'batch_size': 1,        # Reduced for 8GB GPU
    'gradient_accumulation': 16,  # Effective batch size = 16
    'learning_rate': 1e-4,
    'warmup_steps': 2000,
    'max_steps': 50000,
    'gradient_clip_norm': 1.0,
    'weight_decay': 0.1,
    'beta1': 0.9,
    'beta2': 0.95,
    'eps': 1e-8
}

# Hardware optimization for 500M
HARDWARE_500M_CONFIG = {
    'use_flash_attention': True,
    'use_gradient_checkpointing': True,
    'mixed_precision': 'fp16',
    'compile_model': True,
    'fused_optimizer': True
}

# Complete 500M configuration
BWR_DNC_500M_FULL_CONFIG = {
    'model': MODEL_500M_CONFIG,
    'memory': MEMORY_500M_CONFIG,
    'training': TRAINING_500M_CONFIG,
    'hardware': HARDWARE_500M_CONFIG
}

def calculate_model_parameters(config):
    """Calculate approximate parameter count for given configuration."""
    vocab_size = config['vocab_size']
    d_model = config['d_model']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    
    # Embedding parameters
    embedding_params = vocab_size * d_model
    
    # Transformer block parameters (per layer)
    # Attention: Q, K, V, O projections
    attention_params = 4 * (d_model * d_model)
    # Feed-forward: 2 linear layers with 4x expansion
    ff_params = 2 * d_model * (4 * d_model)
    # Layer norms: 2 per layer
    ln_params = 2 * d_model
    
    layer_params = attention_params + ff_params + ln_params
    total_layer_params = n_layers * layer_params
    
    # Output head
    output_params = d_model * vocab_size
    
    # Total
    total_params = embedding_params + total_layer_params + output_params
    
    return total_params

def create_bwr_dnc_500m():
    """
    Create a 500 million parameter BWR-DNC model.
    
    Returns:
        Configured MemoryIntegratedDNC model with ~500M parameters
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.integration import create_integrated_model
    
    return create_integrated_model(
        vocab_size=MODEL_500M_CONFIG['vocab_size'],
        model_config={
            'd_model': MODEL_500M_CONFIG['d_model'],
            'n_layers': MODEL_500M_CONFIG['n_layers'],
            'n_heads': MODEL_500M_CONFIG['n_heads'],
            'max_seq_len': MODEL_500M_CONFIG['max_seq_len'],
            'dropout': MODEL_500M_CONFIG['dropout']
        },
        memory_config=MEMORY_500M_CONFIG
    )

if __name__ == "__main__":
    params = calculate_model_parameters(MODEL_500M_CONFIG)
    print(f"Estimated parameters: {params:,}")
    print(f"Parameter target: 500M")
    print(f"Difference: {abs(params - 500_000_000):,}")
    print(f"Accuracy: {params/500_000_000*100:.1f}% of target")
    print(f"Model size (fp32): {params * 4 / 1024**3:.2f} GB")
    print(f"Model size (fp16): {params * 2 / 1024**3:.2f} GB")
