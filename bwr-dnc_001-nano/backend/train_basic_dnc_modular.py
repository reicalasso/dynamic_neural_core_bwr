"""
Training script for the basic DNC model using modularized code.
"""

import torch
from bwr.models.model_factory import ModelFactory
from bwr.training.trainers.dnc_trainer import DNCTrainer
from bwr.training.config import create_basic_config

def main():
    """Main function to run training."""
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    config = create_basic_config()
    run_name = f"basic_dnc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create model
    model = ModelFactory.create_model("basic", **config['model'])
    
    # Create trainer
    trainer = DNCTrainer(model, config)
    trainer.train(run_name)

if __name__ == "__main__":
    from datetime import datetime
    main()