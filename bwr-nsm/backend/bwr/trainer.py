import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import argparse
import yaml
import wandb
import os
from datetime import datetime
import json
import numpy as np

from .model import NSM
from .dataset import create_toy_dataset, create_curriculum_dataset

# Mock tokenizer for the skeleton
class MockTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
    def __call__(self, text, return_tensors=None):
        return {'input_ids': torch.randint(0, self.vocab_size, (1, 128))}
    def decode(self, ids):
        return f"decoded text for ids of shape {ids.shape}"

class AdvancedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Enable optimizations for RTX GPUs
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.setup_model()
        self.setup_training()
        self.setup_logging()
        
    def setup_model(self):
        """Initialize model with config parameters."""
        model_config = self.config['model']
        self.vocab_size = model_config['vocab_size']
        
        self.model = NSM(
            vocab=self.vocab_size,
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            slots=model_config['slots'],
            max_seq_len=model_config.get('max_seq_len', 2048),
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)
        
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized: {total_params:,} total params, {trainable_params:,} trainable")
        
        self.tokenizer = MockTokenizer(self.vocab_size)
        
    def setup_training(self):
        """Setup optimizer, scheduler, and training components."""
        train_config = self.config['training']
        
        # Optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config['lr'],
            weight_decay=train_config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # Mixed precision scaler for RTX GPUs
        self.scaler = GradScaler() if self.device == 'cuda' else None
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=train_config['epochs'],
            eta_min=train_config['lr'] * 0.1
        )
        
        # Training parameters
        self.epochs = train_config['epochs']
        self.batch_size = train_config['batch_size']
        self.grad_clip = train_config['grad_clip']
        self.accumulation_steps = train_config.get('accumulation_steps', 1)
        self.eval_every = train_config.get('eval_every', 100)
        
    def setup_logging(self):
        """Setup logging and experiment tracking."""
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('project_name', 'bwr-nsm'),
                config=self.config,
                name=self.config.get('run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            )
            
        self.metrics = {'train_loss': [], 'eval_loss': [], 'lr': []}
        
    def create_dataloader(self, task='copy'):
        """Create sophisticated dataloader with curriculum learning."""
        if self.config['training'].get('curriculum_learning', False):
            # Curriculum learning with multiple tasks
            tasks = ['copy', 'lookup', 'long_infill', 'needle_haystack']
            dataset = create_curriculum_dataset(
                tasks=tasks,
                num_samples_per_task=1000,
                seq_len=self.config['model'].get('max_seq_len', 128),
                vocab_size=self.vocab_size
            )
            
            # Custom collate function for mixed tasks
            def collate_fn(batch):
                inputs, targets, task_names = zip(*batch)
                return (
                    torch.stack([torch.tensor(inp, dtype=torch.long) for inp in inputs]),
                    torch.stack([torch.tensor(tgt, dtype=torch.long) for tgt in targets]),
                    task_names
                )
            
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        else:
            # Single task training
            dataset = create_toy_dataset(
                task=task,
                num_samples=2000,
                seq_len=self.config['model'].get('max_seq_len', 128),
                vocab_size=self.vocab_size
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def compute_loss(self, logits, targets, aux_info=None):
        """Compute loss with additional regularization."""
        # Standard cross-entropy loss
        main_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0  # Ignore padding tokens
        )
        
        total_loss = main_loss
        loss_components = {'main': main_loss.item()}
        
        # State regularization loss
        if aux_info and 'route_weights' in aux_info:
            route_weights = aux_info['route_weights']
            # Encourage balanced routing across memory levels
            route_entropy = -torch.sum(route_weights * torch.log(route_weights + 1e-8), dim=-1)
            route_loss = -route_entropy.mean() * 0.01  # Small weight
            total_loss += route_loss
            loss_components['route'] = route_loss.item()
        
        # Memory utilization loss
        if aux_info and 'attention_maps' in aux_info:
            # Encourage diverse memory access patterns
            attn_maps = aux_info['attention_maps']
            if attn_maps:
                # This is a simplified version - would need proper attention map processing
                memory_diversity_loss = torch.tensor(0.0, device=self.device)
                total_loss += memory_diversity_loss * 0.005
                loss_components['memory_div'] = memory_diversity_loss.item()
        
        return total_loss, loss_components
    
    def train_step(self, batch):
        """Single training step with mixed precision."""
        if len(batch) == 3:  # Curriculum learning
            inputs, targets, _ = batch
        else:  # Single task
            inputs, targets = batch
            
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with autocast(enabled=self.scaler is not None):
            logits, aux_info = self.model(inputs, return_state_info=True)
            loss, loss_components = self.compute_loss(logits, targets, aux_info)
            loss = loss / self.accumulation_steps
        
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.accumulation_steps, loss_components
    
    def eval_step(self, batch):
        """Evaluation step."""
        if len(batch) == 3:
            inputs, targets, _ = batch
        else:
            inputs, targets = batch
            
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            with autocast(enabled=self.scaler is not None):
                logits, aux_info = self.model(inputs, return_state_info=True)
                loss, loss_components = self.compute_loss(logits, targets, aux_info)
        
        # Compute accuracy
        predictions = torch.argmax(logits, dim=-1)
        mask = targets != 0  # Ignore padding
        accuracy = (predictions == targets)[mask].float().mean()
        
        return loss.item(), accuracy.item(), loss_components
    
    def train(self, run_name):
        """Main training loop with advanced features."""
        print(f"Starting training for run: {run_name}")
        
        # Create dataloaders
        train_loader = self.create_dataloader('copy')  # Can be changed to other tasks
        eval_loader = self.create_dataloader('copy')  # Separate eval set in real implementation
        
        best_loss = float('inf')
        step = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Training step
                loss, loss_components = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                step += 1
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                # Logging
                if step % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{self.epochs}, Step {step}, Loss: {loss:.4f}, LR: {current_lr:.6f}")
                    
                    if self.config.get('use_wandb', False):
                        log_dict = {
                            'train/loss': loss,
                            'train/lr': current_lr,
                            'train/step': step
                        }
                        log_dict.update({f'train/{k}': v for k, v in loss_components.items()})
                        wandb.log(log_dict)
                
                # Evaluation
                if step % self.eval_every == 0:
                    self.model.eval()
                    eval_loss, eval_acc, eval_components = self.evaluate(eval_loader)
                    
                    print(f"Eval - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")
                    
                    if self.config.get('use_wandb', False):
                        eval_log = {
                            'eval/loss': eval_loss,
                            'eval/accuracy': eval_acc,
                            'eval/step': step
                        }
                        eval_log.update({f'eval/{k}': v for k, v in eval_components.items()})
                        wandb.log(eval_log)
                    
                    # Save best model
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        self.save_checkpoint(run_name, epoch, step, eval_loss)
                    
                    self.model.train()
            
            # End of epoch
            avg_loss = epoch_loss / num_batches
            self.scheduler.step()
            
            print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
            self.metrics['train_loss'].append(avg_loss)
        
        print("Training completed!")
        
    def evaluate(self, eval_loader):
        """Comprehensive evaluation."""
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        all_loss_components = {}
        
        with torch.no_grad():
            for batch in eval_loader:
                loss, accuracy, loss_components = self.eval_step(batch)
                total_loss += loss
                total_accuracy += accuracy
                num_batches += 1
                
                # Aggregate loss components
                for k, v in loss_components.items():
                    if k not in all_loss_components:
                        all_loss_components[k] = []
                    all_loss_components[k].append(v)
        
        # Average results
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_components = {k: np.mean(v) for k, v in all_loss_components.items()}
        
        return avg_loss, avg_accuracy, avg_components
    
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
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = f"{checkpoint_dir}/checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

def train(config_path, run_name):
    """Main training function."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = AdvancedTrainer(config)
    trainer.train(run_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--run-name', type=str, default='bwr_run', help='Name for the training run')
    args = parser.parse_args()
    train(args.config, args.run_name)
