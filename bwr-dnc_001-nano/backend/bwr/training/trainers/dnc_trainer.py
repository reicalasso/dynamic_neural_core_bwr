"""
DNC trainer implementation.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os
from datetime import datetime
from .base_trainer import BaseTrainer
from ..datasets.copy_dataset import SimpleCopyDataset, collate_fn

class DNCTrainer(BaseTrainer):
    """Trainer for the DNC model."""
    
    def __init__(self, model, config):
        self.model = model
        super().__init__(config)
    
    def setup_model(self):
        """Initialize model with config parameters."""
        # Model is already initialized in __init__
        self.model = self.model.to(self.device)
        
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