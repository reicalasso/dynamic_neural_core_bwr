"""
Simple curriculum learning for the minimal enhanced DNC model.

This script implements a basic curriculum learning approach where we start
with very simple tasks and gradually increase complexity.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from datetime import datetime

# Add the backend directory to the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

from minimal_dnc import MinimalEnhancedDNC

class SimpleCopyDataset(Dataset):
    """A simple dataset for copy tasks of varying difficulty."""
    
    def __init__(self, num_samples=1000, max_seq_len=16, vocab_size=50, difficulty=1):
        """
        Args:
            num_samples: Number of samples to generate
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
            difficulty: Difficulty level (1-10, where 1 is easiest)
        """
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.difficulty = difficulty
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate copy task data with progressive difficulty."""
        data = []
        
        # Calculate sequence length based on difficulty
        # Difficulty 1: length 2, Difficulty 10: length max_seq_len
        seq_len = max(2, min(self.max_seq_len, int(self.difficulty * self.max_seq_len / 10)))
        
        for _ in range(self.num_samples):
            # Generate random sequence
            sequence = torch.randint(1, self.vocab_size-2, (seq_len,))
            
            # Add start and end tokens
            start_token = torch.tensor([0])  # Start token
            end_token = torch.tensor([self.vocab_size-1])  # End token
            
            # Input: start_token + sequence
            # Target: sequence + end_token
            input_seq = torch.cat([start_token, sequence])
            target_seq = torch.cat([sequence, end_token])
            
            data.append((input_seq, target_seq))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Collate function to handle variable length sequences."""
    inputs, targets = zip(*batch)
    
    # Pad sequences to the same length
    max_input_len = max([inp.shape[0] for inp in inputs])
    max_target_len = max([tgt.shape[0] for tgt in targets])
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        # Pad input
        input_padding = torch.zeros(max_input_len - inp.shape[0], dtype=torch.long)
        padded_input = torch.cat([inp, input_padding])
        padded_inputs.append(padded_input)
        
        # Pad target
        target_padding = torch.zeros(max_target_len - tgt.shape[0], dtype=torch.long)
        padded_target = torch.cat([tgt, target_padding])
        padded_targets.append(padded_target)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)

class CurriculumTrainer:
    """Simple curriculum trainer for the minimal enhanced DNC."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.setup_model()
    
    def setup_model(self):
        """Initialize model with config parameters."""
        model_config = self.config['model']
        
        self.model = MinimalEnhancedDNC(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            max_seq_len=model_config['max_seq_len']
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized: {total_params:,} total params, {trainable_params:,} trainable")
    
    def compute_loss(self, logits, targets):
        """Compute loss."""
        # Standard cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1  # Don't ignore any tokens
        )
        return loss
    
    def train_stage(self, difficulty, epochs, lr):
        """Train on a specific difficulty level."""
        print(f"\n=== Training Stage: Difficulty {difficulty}/10 ===")
        
        # Create dataset for this difficulty level
        train_dataset = SimpleCopyDataset(
            num_samples=500,  # Smaller dataset for faster iteration
            max_seq_len=self.config['model']['max_seq_len'] // 2,
            vocab_size=self.config['model']['vocab_size'] - 2,
            difficulty=difficulty
        )
        
        eval_dataset = SimpleCopyDataset(
            num_samples=100,
            max_seq_len=self.config['model']['max_seq_len'] // 2,
            vocab_size=self.config['model']['vocab_size'] - 2,
            difficulty=difficulty
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Setup optimizer and scheduler for this stage
        optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, epochs // 3),
            gamma=0.5
        )
        
        best_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                logits, _ = self.model(inputs)
                
                # Compute loss
                loss = self.compute_loss(logits, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # End of epoch
            avg_loss = epoch_loss / num_batches
            scheduler.step()
            
            # Evaluate
            eval_loss, eval_acc = self.evaluate(eval_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, "
                  f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc*100:.2f}%, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model for this stage
            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save_checkpoint(f"curriculum_difficulty_{difficulty}", epoch, eval_loss, eval_acc)
        
        return best_loss
    
    def evaluate(self, dataloader):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                logits, _ = self.model(inputs)
                loss = self.compute_loss(logits, targets)
                
                # Compute accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = targets != -1  # Don't ignore any tokens
                accuracy = (predictions == targets)[mask].float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches, total_accuracy / num_batches
    
    def train_curriculum(self):
        """Train using curriculum learning."""
        print("Starting curriculum learning with minimal enhanced DNC...")
        
        # Start with easiest difficulty and gradually increase
        for difficulty in range(1, 11):  # 1 to 10
            # Adjust learning rate and epochs based on difficulty
            lr = self.config['training']['lr'] * (0.9 ** (difficulty - 1))  # Gradually reduce LR
            epochs = max(3, 10 - difficulty)  # Fewer epochs for easier tasks
            
            # Train at this difficulty level
            best_loss = self.train_stage(difficulty, epochs, lr)
            print(f"Best loss at difficulty {difficulty}: {best_loss:.4f}")
            
            # Test generation at this stage
            self.test_generation()
        
        print("Curriculum learning completed!")
    
    def test_generation(self):
        """Test the model's text generation capabilities."""
        self.model.eval()
        
        # Create a simple test input
        start_token = 0
        test_input = torch.tensor([[start_token, 1, 2]], dtype=torch.long).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(test_input, max_length=8)
        
        print(f"  Generation test - Input: {test_input.cpu().tolist()}")
        print(f"  Generated: {generated.cpu().tolist()}")
        
        self.model.train()
    
    def save_checkpoint(self, stage_name, epoch, eval_loss, eval_acc):
        """Save model checkpoint."""
        checkpoint_dir = f"checkpoints/minimal_{stage_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'config': self.config
        }
        
        checkpoint_path = f"{checkpoint_dir}/best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")

def create_minimal_config():
    """Create configuration for minimal enhanced DNC training."""
    return {
        'model': {
            'vocab_size': 32,           # Smaller vocabulary
            'd_model': 128,             # Model dimension
            'n_layers': 4,              # Number of layers
            'n_heads': 8,               # Attention heads
            'max_seq_len': 32           # Maximum sequence length
        },
        'training': {
            'batch_size': 32,
            'lr': 5e-4,                 # Slightly higher learning rate
            'weight_decay': 0.01,
            'grad_clip': 1.0
        }
    }

def main():
    """Main function to run curriculum learning."""
    config = create_minimal_config()
    
    trainer = CurriculumTrainer(config)
    trainer.train_curriculum()

if __name__ == "__main__":
    main()