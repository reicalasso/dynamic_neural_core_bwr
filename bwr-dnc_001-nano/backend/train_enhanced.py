"""
Training script for the Enhanced DNC model.

This script trains the enhanced DNC model with hierarchical memory
using curriculum learning and detailed monitoring.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from datetime import datetime
import json

# Add the backend directory to the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

from enhanced_dnc import EnhancedDNC

class ProgressiveCopyDataset(Dataset):
    """A dataset for progressive copy tasks of increasing difficulty."""
    
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

class EnhancedDNCTrainer:
    """Trainer for the enhanced DNC model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.setup_model()
        self.training_log = []
    
    def setup_model(self):
        """Initialize model with config parameters."""
        model_config = self.config['model']
        
        self.model = EnhancedDNC(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            num_memory_levels=model_config['num_memory_levels'],
            base_memory_slots=model_config['base_memory_slots'],
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
    
    def train_stage(self, difficulty, epochs, lr, log_interval=10):
        """Train on a specific difficulty level."""
        print(f"\n=== Training Stage: Difficulty {difficulty}/10 ===")
        
        # Create dataset for this difficulty level
        train_dataset = ProgressiveCopyDataset(
            num_samples=500,  # Smaller dataset for faster iteration
            max_seq_len=self.config['model']['max_seq_len'] // 2,
            vocab_size=self.config['model']['vocab_size'] - 2,
            difficulty=difficulty
        )
        
        eval_dataset = ProgressiveCopyDataset(
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
        stage_log = {
            'difficulty': difficulty,
            'epochs': [],
            'best_eval_loss': float('inf'),
            'best_eval_acc': 0.0
        }
        
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
            
            # Log epoch results
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc,
                'lr': optimizer.param_groups[0]['lr']
            }
            stage_log['epochs'].append(epoch_log)
            
            # Print progress
            if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, "
                      f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc*100:.2f}%, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model for this stage
            if eval_loss < best_loss:
                best_loss = eval_loss
                stage_log['best_eval_loss'] = eval_loss
                stage_log['best_eval_acc'] = eval_acc
                self.save_checkpoint(f"enhanced_difficulty_{difficulty}", epoch, eval_loss, eval_acc)
        
        self.training_log.append(stage_log)
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
        print("Starting curriculum learning with enhanced DNC...")
        
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
        
        # Save training log
        self.save_training_log()
        print("Enhanced curriculum learning completed!")
    
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
        checkpoint_dir = f"checkpoints/enhanced_{stage_name}"
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
    
    def save_training_log(self):
        """Save the complete training log."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = f"{log_dir}/enhanced_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert tensor objects to serializable format
        serializable_log = []
        for stage in self.training_log:
            serializable_stage = stage.copy()
            serializable_stage['epochs'] = []
            for epoch in stage['epochs']:
                serializable_epoch = {}
                for key, value in epoch.items():
                    if isinstance(value, (int, float)):
                        serializable_epoch[key] = value
                    else:
                        serializable_epoch[key] = str(value)
                serializable_stage['epochs'].append(serializable_epoch)
            serializable_log.append(serializable_stage)
        
        with open(log_path, 'w') as f:
            json.dump(serializable_log, f, indent=2)
        
        print(f"Training log saved to: {log_path}")

def create_enhanced_config():
    """Create configuration for enhanced DNC training."""
    return {
        'model': {
            'vocab_size': 32,           # Smaller vocabulary
            'd_model': 256,             # Model dimension
            'n_layers': 4,              # Number of layers
            'n_heads': 8,               # Attention heads
            'num_memory_levels': 3,     # Hierarchical memory levels
            'base_memory_slots': 128,   # Base memory slots
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
    """Main function to run enhanced DNC training."""
    config = create_enhanced_config()
    
    trainer = EnhancedDNCTrainer(config)
    trainer.train_curriculum()

if __name__ == "__main__":
    main()