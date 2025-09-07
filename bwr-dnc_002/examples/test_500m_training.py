#!/usr/bin/env python3
"""
BWR-DNC 500M Training Test

Quick test of training functionality for 500M model.
"""

import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.bwr_dnc_500m import create_bwr_dnc_500m, MODEL_500M_CONFIG, TRAINING_500M_CONFIG

def test_training_step():
    """Test a single training step with optimized settings"""
    print("Testing BWR-DNC 500M Training Step...")
    print("=" * 50)
    
    # Clear GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        model = create_bwr_dnc_500m()
        model.train()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_500M_CONFIG['learning_rate'])
        
        # Create small training batch
        batch_size = TRAINING_500M_CONFIG['batch_size']  # Should be 1 now
        seq_len = 64  # Smaller sequence for training test
        
        input_ids = torch.randint(0, MODEL_500M_CONFIG['vocab_size'], 
                                (batch_size, seq_len), device=device)
        targets = torch.randint(0, MODEL_500M_CONFIG['vocab_size'], 
                              (batch_size, seq_len), device=device)
        
        print(f"Training with batch_size={batch_size}, seq_len={seq_len}")
        
        # Training step
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Calculate loss
        loss = torch.nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        step_time = time.time() - start_time
        
        print(f"✓ Training step successful!")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Step time: {step_time:.3f}s")
        
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU memory used: {memory_used:.2f} GB")
        
        print("\n✅ BWR-DNC 500M training is functional!")
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False

if __name__ == "__main__":
    success = test_training_step()
    exit(0 if success else 1)
