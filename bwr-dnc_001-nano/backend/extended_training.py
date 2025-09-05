"""
Extended training for high difficulty levels.

This script continues training on the more difficult tasks for longer periods.
"""

import torch
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bwr'))

from train_curriculum import CurriculumTrainer, create_curriculum_config

def extended_training():
    """Continue training on higher difficulty levels."""
    config = create_curriculum_config()
    
    # Create trainer
    trainer = CurriculumTrainer(config)
    
    # Load the best model from difficulty 10
    checkpoint_path = "checkpoints/curriculum_curriculum_difficulty_10/best_model.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print("No checkpoint found, starting from scratch")
    
    # Continue training at difficulty 10 for more epochs
    print("\n=== Extended Training: Difficulty 10 (20 epochs) ===")
    best_loss = trainer.train_stage(difficulty=10, epochs=20, lr=1e-4)
    print(f"Best loss after extended training: {best_loss:.4f}")
    
    # Test generation
    trainer.test_generation()
    
    # Save final model
    trainer.save_checkpoint("final_extended", 0, best_loss)
    print("Extended training completed!")

if __name__ == "__main__":
    extended_training()