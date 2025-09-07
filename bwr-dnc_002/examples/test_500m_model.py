#!/usr/bin/env python3
"""
BWR-DNC 500M Model Test

Test the 500 million parameter BWR-DNC model.
This script tests model creation, parameter counting, and training functionality.
"""

import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.bwr_dnc_500m import create_bwr_dnc_500m, MODEL_500M_CONFIG, TRAINING_500M_CONFIG
from utils import calculate_model_params, format_memory_size

def test_500m_model_creation():
    """Test creating the 500M parameter model"""
    print("Creating BWR-DNC 500M Model...")
    print("=" * 50)
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory Available: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 8:
            print("✅ GPU memory should be sufficient for 500M model")
        else:
            print("⚠️  Warning: Limited GPU memory. Consider CPU training.")
    
    # Create model
    print("Creating model...")
    model = create_bwr_dnc_500m()
    
    # Calculate actual parameters
    actual_params = calculate_model_params(model)
    model_size_gb = actual_params * 4 / 1024**3  # fp32
    model_size_gb_fp16 = actual_params * 2 / 1024**3  # fp16
    
    print(f"✓ Model created successfully!")
    print(f"  Target parameters: 500,000,000")
    print(f"  Actual parameters: {actual_params:,}")
    print(f"  Accuracy: {actual_params/500_000_000*100:.1f}% of 500M")
    print(f"  Model size (fp32): {model_size_gb:.2f} GB")
    print(f"  Model size (fp16): {model_size_gb_fp16:.2f} GB")
    print()
    
    return model

def test_500m_model_forward():
    """Test forward pass with different input sizes"""
    print("Testing Forward Pass...")
    print("=" * 50)
    
    model = create_bwr_dnc_500m()
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Test with different batch and sequence sizes
    test_cases = [
        (1, 32),   # Small test
        (2, 64),   # Medium test
        (4, 128),  # Larger test
    ]
    
    for batch_size, seq_len in test_cases:
        try:
            input_ids = torch.randint(0, MODEL_500M_CONFIG['vocab_size'], 
                                    (batch_size, seq_len), device=device)
            
            start_time = time.time()
            with torch.no_grad():
                output, info = model(input_ids)
            
            forward_time = time.time() - start_time
            
            print(f"✓ Forward pass successful!")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Forward time: {forward_time:.3f}s")
            
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU memory used: {memory_used:.2f} GB")
            
            print()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Out of memory for batch_size={batch_size}, seq_len={seq_len}")
                break
            else:
                print(f"❌ Forward pass failed: {e}")
                return False
    
    return True

def test_500m_generation():
    """Test text generation with 500M model"""
    print("Testing Text Generation...")
    print("=" * 50)
    
    try:
        model = create_bwr_dnc_500m()
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Test generation with different prompts
        prompts = [
            [1, 2, 3, 4, 5],           # Simple numeric prompt
            [10, 20, 30, 40, 50],      # Different pattern
            [100, 200, 300],           # Shorter prompt
        ]
        
        for i, prompt_tokens in enumerate(prompts):
            prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    prompt,
                    max_length=20,
                    temperature=0.8,
                    use_memory=True
                )
            
            gen_time = time.time() - start_time
            
            print(f"✓ Generation {i+1} successful!")
            print(f"  Input: {prompt[0].tolist()}")
            print(f"  Generated: {output[0].tolist()}")
            print(f"  Generation time: {gen_time:.3f}s")
            print(f"  Tokens/sec: {len(output[0])/gen_time:.1f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def test_500m_training_step():
    """Test a single training step"""
    print("Testing Training Step...")
    print("=" * 50)
    
    try:
        model = create_bwr_dnc_500m()
        model.train()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_500M_CONFIG['learning_rate'])
        
        # Create sample training data
        batch_size = TRAINING_500M_CONFIG['batch_size']
        seq_len = 128
        
        input_ids = torch.randint(0, MODEL_500M_CONFIG['vocab_size'], 
                                (batch_size, seq_len), device=device)
        targets = torch.randint(0, MODEL_500M_CONFIG['vocab_size'], 
                              (batch_size, seq_len), device=device)
        
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
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage patterns"""
    print("Testing Memory Usage...")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return True
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Create model
    model = create_bwr_dnc_500m()
    model = model.to('cuda')
    
    print(f"After model loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4]
    seq_len = 128
    
    for batch_size in batch_sizes:
        try:
            input_ids = torch.randint(0, MODEL_500M_CONFIG['vocab_size'], 
                                    (batch_size, seq_len), device='cuda')
            
            with torch.no_grad():
                output, _ = model(input_ids)
            
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"Batch size {batch_size}: {memory_used:.2f} GB")
            
            # Clean up
            del input_ids, output
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: Out of memory")
                break
    
    print()
    return True

def main():
    """Run 500M model tests"""
    print("BWR-DNC 500M Parameter Model Test")
    print("=" * 60)
    print()
    
    try:
        # Test model creation
        test_500m_model_creation()
        
        # Test forward pass
        forward_success = test_500m_model_forward()
        
        # Test generation
        if forward_success:
            generation_success = test_500m_generation()
        else:
            generation_success = False
        
        # Test training step
        if forward_success:
            training_success = test_500m_training_step()
        else:
            training_success = False
        
        # Test memory usage
        test_memory_usage()
        
        print("500M Model Test Summary")
        print("=" * 60)
        print(f"✓ Model Creation: Success")
        print(f"{'✓' if forward_success else '❌'} Forward Pass: {'Success' if forward_success else 'Failed'}")
        print(f"{'✓' if generation_success else '❌'} Text Generation: {'Success' if generation_success else 'Failed'}")
        print(f"{'✓' if training_success else '❌'} Training Step: {'Success' if training_success else 'Failed'}")
        print()
        
        if all([forward_success, generation_success, training_success]):
            print("🎉 BWR-DNC 500M model is fully functional!")
            print("Ready for training and deployment on current hardware.")
        else:
            print("⚠️  Some tests failed. Check GPU memory and configuration.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
