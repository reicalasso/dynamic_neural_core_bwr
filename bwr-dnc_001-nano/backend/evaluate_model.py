"""
Evaluation script for the trained DNC model.

This script loads a trained model and evaluates its performance on a test set.
"""

import torch
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

from basic_dnc import BasicDNC
from train_basic_dnc import SimpleCopyDataset, collate_fn
from torch.utils.data import DataLoader

def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with config from checkpoint
    config = checkpoint['config']
    model = BasicDNC(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        slots=config['model']['slots'],
        max_seq_len=config['model']['max_seq_len']
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model, config

def evaluate_model(model, dataloader, device):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, _ = model(inputs)
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = targets != -1  # Don't ignore any tokens
            accuracy = (predictions == targets)[mask].float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy

def test_generation(model, vocab_size, device):
    """Test the model's text generation capabilities."""
    model.eval()
    
    # Create a simple test input
    start_token = 0
    test_input = torch.tensor([[start_token, 1, 2, 3]], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(test_input, max_length=10)
    
    print(f"Input: {test_input.cpu().tolist()}")
    print(f"Generated: {generated.cpu().tolist()}")
    
    return generated

def main():
    """Main evaluation function."""
    # Check if checkpoint path is provided
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Create test dataset
    test_dataset = SimpleCopyDataset(
        num_samples=200,
        seq_len=config['model']['max_seq_len'] // 2,
        vocab_size=config['model']['vocab_size'] - 2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate
    print("Evaluating model...")
    loss, accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Test generation
    print("\nTesting generation...")
    generated = test_generation(model, config['model']['vocab_size'], device)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()