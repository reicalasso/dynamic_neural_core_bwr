"""
Base trainer classes for DNC training.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

class BaseTrainer:
    """Base class for all DNC trainers."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_training()
    
    def setup_model(self):
        """Initialize model with config parameters."""
        raise NotImplementedError("Subclasses must implement setup_model method")
    
    def setup_training(self):
        """Setup optimizer and training components."""
        raise NotImplementedError("Subclasses must implement setup_training method")
    
    def compute_loss(self, logits, targets):
        """Compute loss."""
        raise NotImplementedError("Subclasses must implement compute_loss method")
    
    def train_step(self, batch):
        """Single training step."""
        raise NotImplementedError("Subclasses must implement train_step method")
    
    def evaluate(self, dataloader):
        """Evaluate model."""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def train(self, run_name):
        """Main training loop."""
        raise NotImplementedError("Subclasses must implement train method")
    
    def save_checkpoint(self, run_name, epoch, step, eval_loss):
        """Save model checkpoint."""
        raise NotImplementedError("Subclasses must implement save_checkpoint method")