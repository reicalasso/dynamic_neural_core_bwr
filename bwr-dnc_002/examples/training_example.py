"""
BWR-DNC 002: Training Example

This example demonstrates how to train the BWR-DNC model on a simple dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.integration import create_integrated_model
from utils import (
    Config, MetricsTracker, EarlyStopping, 
    get_device, set_seed, save_checkpoint, format_time
)
import time


def create_synthetic_dataset(vocab_size: int, seq_len: int, num_samples: int):
    """
    Create a synthetic dataset for training.
    
    Args:
        vocab_size: Vocabulary size
        seq_len: Sequence length
        num_samples: Number of samples
        
    Returns:
        DataLoader for training
    """
    # Create random sequences
    data = torch.randint(1, vocab_size, (num_samples, seq_len))
    
    # Create labels (next token prediction)
    labels = torch.cat([data[:, 1:], torch.randint(1, vocab_size, (num_samples, 1))], dim=1)
    
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=8, shuffle=True)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    metrics_tracker: MetricsTracker
):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        metrics_tracker: Metrics tracker
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, metadata = model(input_ids)
        
        # Compute loss
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1
        
        # Update metrics tracker
        metrics_tracker.update({
            'train_loss': batch_loss,
            'write_strength': metadata.get('write_strength', 0.0)
        })
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx:3d}: loss = {batch_loss:.4f}, "
                  f"write_strength = {metadata.get('write_strength', 0.0):.3f}")
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    """Main training function."""
    print("BWR-DNC 002: Training Example")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Training configuration
    config = Config({
        'model': {
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 8,
            'max_seq_len': 512,
            'dropout': 0.1
        },
        'memory': {
            'memory_slots': [512, 256, 128],
            'memory_integration_layers': [2, 3]
        },
        'training': {
            'vocab_size': 5000,
            'seq_len': 128,
            'num_samples': 1000,
            'batch_size': 8,
            'learning_rate': 0.001,
            'num_epochs': 10,
            'patience': 3
        }
    })
    
    # Create model
    print("\nCreating model...")
    model = create_integrated_model(
        vocab_size=config.get('training.vocab_size'),
        model_config=config.get('model'),
        memory_config=config.get('memory')
    )
    
    model = model.to(device)
    
    # Print model info
    from utils import count_parameters, get_model_size_mb
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {model_size:.1f} MB")
    
    # Create dataset
    print("\nCreating synthetic dataset...")
    train_loader = create_synthetic_dataset(
        config.get('training.vocab_size'),
        config.get('training.seq_len'),
        config.get('training.num_samples')
    )
    
    val_loader = create_synthetic_dataset(
        config.get('training.vocab_size'),
        config.get('training.seq_len'),
        config.get('training.num_samples') // 5  # Smaller validation set
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training.learning_rate'),
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('training.num_epochs')
    )
    
    # Metrics tracking and early stopping
    metrics_tracker = MetricsTracker(window_size=50)
    early_stopping = EarlyStopping(patience=config.get('training.patience'))
    
    print("\nStarting training...")
    print("=" * 40)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config.get('training.num_epochs')):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{config.get('training.num_epochs')}")
        print("-" * 30)
        
        # Train
        avg_train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, metrics_tracker
        )
        
        # Validate
        avg_val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Update metrics
        metrics_tracker.update({
            'val_loss': avg_val_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        # Get memory stats
        sample_input = torch.randint(1, config.get('training.vocab_size'), (1, 32)).to(device)
        with torch.no_grad():
            _, metadata = model(sample_input, return_memory_stats=True)
            memory_stats = metadata['memory_stats']
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Memory Utilization: {memory_stats['utilization']:.1%}")
        print(f"  Epoch Time: {format_time(epoch_time)}")
        
        # Save checkpoint if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = project_root / "checkpoints" / "best_model.pt"
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_val_loss,
                checkpoint_path=checkpoint_path,
                metadata={
                    'config': config.to_dict(),
                    'memory_stats': memory_stats
                }
            )
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 40)
    print("Training completed!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final metrics
    final_averages = metrics_tracker.get_averages()
    print("\nFinal Metrics:")
    for name, value in final_averages.items():
        print(f"  {name}: {value:.4f}")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    test_input = torch.randint(1, 100, (1, 10)).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            test_input,
            max_length=30,
            temperature=0.8,
            use_memory=True
        )
    
    print(f"Generated sequence: {generated[0].tolist()}")
    
    print("\nTraining example completed successfully!")


if __name__ == '__main__':
    main()
