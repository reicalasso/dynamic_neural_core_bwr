"""
BWR-DNC 002: Basic Usage Example

This example demonstrates basic usage of the BWR-DNC model for text generation.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.integration import create_integrated_model
from utils import Config, get_device, set_seed


def main():
    """Main example function."""
    print("BWR-DNC 002: Basic Usage Example")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model configuration
    model_config = {
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'max_seq_len': 2048,
        'dropout': 0.1
    }
    
    memory_config = {
        'memory_slots': [1024, 512, 256],
        'memory_integration_layers': [4, 5]  # Integrate memory at layers 4 and 5
    }
    
    # Create integrated model
    print("\nCreating BWR-DNC model...")
    model = create_integrated_model(
        vocab_size=50000,  # Typical vocabulary size
        model_config=model_config,
        memory_config=memory_config
    )
    
    model = model.to(device)
    
    # Print model information
    from utils import count_parameters, get_model_size_mb
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size:.1f} MB")
    
    # Example input (random tokens for demonstration)
    print("\nGenerating text...")
    batch_size = 1
    initial_length = 20
    max_length = 50
    
    # Create some initial tokens
    input_ids = torch.randint(1, 1000, (batch_size, initial_length)).to(device)
    
    # Generate with memory
    print("Generating with external memory...")
    model.eval()
    with torch.no_grad():
        generated_with_memory = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.8,
            top_p=0.9,
            use_memory=True
        )
    
    # Generate without memory for comparison
    print("Generating without external memory...")
    with torch.no_grad():
        generated_without_memory = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.8,
            top_p=0.9,
            use_memory=False
        )
    
    print(f"\nGeneration complete!")
    print(f"Input length: {initial_length}")
    print(f"Generated length: {max_length}")
    print(f"With memory - tokens: {generated_with_memory[0].tolist()}")
    print(f"Without memory - tokens: {generated_without_memory[0].tolist()}")
    
    # Get memory statistics
    print("\nMemory Statistics:")
    _, metadata = model(input_ids, return_memory_stats=True)
    memory_stats = metadata['memory_stats']
    
    for level_name, stats in memory_stats.items():
        if level_name.startswith('level_'):
            print(f"{level_name}: {stats['active_slots']}/{stats['slots']} slots active "
                  f"(avg salience: {stats['avg_salience']:.3f})")
    
    print(f"Total utilization: {memory_stats['utilization']:.1%}")
    
    # Demonstrate memory visualization
    print("\nMemory Visualization Data:")
    viz_data = model.get_memory_visualization()
    for level_name, data in viz_data.items():
        print(f"{level_name}: {data['active_slots']} active slots")
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()
