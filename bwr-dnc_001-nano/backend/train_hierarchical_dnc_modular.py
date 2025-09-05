"""
Training script for the hierarchical DNC model using modularized code.
"""

import torch
from bwr.models.model_factory import ModelFactory
from bwr.training.trainers.dnc_trainer import DNCTrainer
from bwr.training.config import create_hierarchical_config

def main():
    """Main function to run training."""
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    config = create_hierarchical_config()
    run_name = f"hierarchical_dnc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create model
    model = ModelFactory.create_model("hierarchical", **config['model'])
    
    # Create trainer
    trainer = DNCTrainer(model, config)
    trainer.train(run_name)

if __name__ == "__main__":
    from datetime import datetime
    main()