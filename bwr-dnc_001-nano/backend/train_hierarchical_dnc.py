"""
Training script for the hierarchical DNC model.

This script demonstrates how to train the hierarchical DNC model on a simple task.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
from datetime import datetime
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

from hierarchical_dnc import HierarchicalDNC

class SimpleCopyDataset(Dataset):
    """A simple dataset for the copy task."""
    
    def __init__(self, num_samples=1000, seq_len=16, vocab_size=50):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate copy task data."""
        data = []
        for _ in range(self.num_samples):
            # Generate random sequence with smaller vocabulary
            sequence = torch.randint(1, self.vocab_size, (self.seq_len,))
            # Add start and end tokens
            start_token = torch.tensor([self.vocab_size])  # Special start token
            end_token = torch.tensor([self.vocab_size + 1])  # Special end token
            
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

class HierarchicalDNCTrainer:
    """Trainer for the hierarchical DNC model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.setup_model()
        self.setup_training()
    
    def setup_model(self):
        """Initialize model with config parameters."""
        model_config = self.config['model']
        
        self.model = HierarchicalDNC(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            slots=model_config['slots'],
            levels=model_config['levels'],
            max_seq_len=model_config['max_seq_len']
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized: {total_params:,} total params, {trainable_params:,} trainable")
    
    def setup_training(self):
        """Setup optimizer and training components."""
        train_config = self.config['training']
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config['lr'],
            weight_decay=train_config.get('weight_decay', 0.01)
        )
        
        # Scheduler with slower decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config['epochs'] * 100,  # Scale with steps instead of epochs
            eta_min=train_config['lr'] * 0.1
        )
        
        # Training parameters
        self.epochs = train_config['epochs']
        self.batch_size = train_config['batch_size']
        self.grad_clip = train_config.get('grad_clip', 1.0)
    
    def compute_loss(self, logits, targets):
        """Compute loss."""
        # Standard cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0  # Ignore padding tokens
        )
        return loss
    
    def train_step(self, batch):
        """Single training step."""
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Forward pass
        logits, _ = self.model(inputs, return_memory_info=True)
        
        # Compute loss
        loss = self.compute_loss(logits, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        
        return loss.item()
    
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
                
                logits, _ = self.model(inputs, return_memory_info=True)
                loss = self.compute_loss(logits, targets)
                
                # Compute accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = targets != 0  # Ignore padding
                accuracy = (predictions == targets)[mask].float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches, total_accuracy / num_batches
    
    def train(self, run_name):
        """Main training loop."""
        print(f"Starting training for run: {run_name}")
        
        # Create datasets
        train_dataset = SimpleCopyDataset(
            num_samples=2000,
            seq_len=self.config['model']['max_seq_len'] // 2,
            vocab_size=self.config['model']['vocab_size'] - 2  # Reserve 2 tokens for start/end
        )
        
        eval_dataset = SimpleCopyDataset(
            num_samples=200,
            seq_len=self.config['model']['max_seq_len'] // 2,
            vocab_size=self.config['model']['vocab_size'] - 2
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        best_loss = float('inf')
        step = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Training step
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                step += 1
                
                # Logging
                if step % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{self.epochs}, Step {step}, Loss: {loss:.4f}, LR: {current_lr:.6f}")
                
                # Evaluation
                if step % 100 == 0:
                    eval_loss, eval_acc = self.evaluate(eval_loader)
                    print(f"Eval - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")
                    
                    # Save best model
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        self.save_checkpoint(run_name, epoch, step, eval_loss)
            
            # End of epoch
            avg_loss = epoch_loss / num_batches
            # Only step scheduler every few epochs to slow down decay
            if (epoch + 1) % 3 == 0:
                self.scheduler.step()
            
            print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
            
            # Evaluate at the end of each epoch
            eval_loss, eval_acc = self.evaluate(eval_loader)
            print(f"Epoch {epoch+1} Eval - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")
        
        print("Training completed!")
    
    def save_checkpoint(self, run_name, epoch, step, eval_loss):
        """Save model checkpoint."""
        checkpoint_dir = f"checkpoints/{run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'eval_loss': eval_loss,
            'config': self.config
        }
        
        checkpoint_path = f"{checkpoint_dir}/checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

def create_config():
    """Create configuration for training."""
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

def main():
    """Main function to run training."""
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    config = create_config()
    run_name = f"hierarchical_dnc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    trainer = HierarchicalDNCTrainer(config)
    trainer.train(run_name)

if __name__ == "__main__":
    main()