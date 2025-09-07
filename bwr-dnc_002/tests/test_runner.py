"""
BWR-DNC 002: Test Runner

Comprehensive test runner for the BWR-DNC project.
Runs unit tests, integration tests, and performance benchmarks.
"""

import pytest
import torch
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
import unittest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.model import DNC, RMSNorm, MultiHeadAttention, DNCBlock
from core.integration import MemoryIntegratedDNC, create_integrated_model
from memory.state_bank import StateBank, create_hierarchical_memory
from utils import Config, MetricsTracker, count_parameters, get_device


class TestCore(unittest.TestCase):
    """Test cases for core model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = get_device()
        self.batch_size = 2
        self.seq_len = 64
        self.d_model = 256
        self.vocab_size = 1000
        
    def test_rmsnorm(self):
        """Test RMSNorm implementation."""
        norm = RMSNorm(self.d_model)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = norm(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check normalization (mean should be close to 0)
        mean_norm = output.pow(2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(mean_norm, torch.ones_like(mean_norm), atol=1e-5))
    
    def test_multihead_attention(self):
        """Test MultiHeadAttention implementation."""
        n_heads = 8
        attn = MultiHeadAttention(self.d_model, n_heads)
        
        q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.seq_len, self.d_model)
        v = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = attn(q, k, v)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check that output is not nan or inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_dnc_block(self):
        """Test DNC block implementation."""
        n_heads = 8
        block = DNCBlock(self.d_model, n_heads)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that output is different from input (processing occurred)
        self.assertFalse(torch.allclose(output, x))
    
    def test_dnc_model(self):
        """Test full DNC model."""
        model = DNC(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=4,
            n_heads=8
        )
        
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        logits, metadata = model(input_ids)
        
        # Check output shapes
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertIn('hidden_states', metadata)
        
        # Check generation
        generated = model.generate(input_ids[:1, :10], max_length=20)
        self.assertEqual(generated.shape, (1, 20))


class TestMemory(unittest.TestCase):
    """Test cases for memory system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = get_device()
        self.batch_size = 2
        self.seq_len = 32
        self.d_model = 256
        self.memory_slots = [512, 256]
        
    def test_state_bank_creation(self):
        """Test StateBank creation and initialization."""
        memory = StateBank(
            d_model=self.d_model,
            slots_per_level=self.memory_slots
        )
        
        # Check that levels are created
        self.assertEqual(len(memory.levels), len(self.memory_slots))
        
        # Check memory stats
        stats = memory.get_memory_stats()
        self.assertIn('total_slots', stats)
        self.assertEqual(stats['total_slots'], sum(self.memory_slots))
    
    def test_memory_read_write(self):
        """Test memory read and write operations."""
        memory = StateBank(
            d_model=self.d_model,
            slots_per_level=self.memory_slots
        )
        
        # Test read operation
        queries = torch.randn(self.batch_size, self.seq_len, self.d_model)
        reads = memory.read(queries)
        
        self.assertEqual(reads.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Test write operation
        writes = torch.randn(self.batch_size, self.seq_len, self.d_model)
        memory.write(writes)
        
        # Check that salience is updated
        stats_after = memory.get_memory_stats()
        self.assertGreater(stats_after['active_slots'], 0)
    
    def test_hierarchical_memory_creation(self):
        """Test hierarchical memory factory function."""
        memory = create_hierarchical_memory(
            d_model=self.d_model,
            base_slots=1024,
            levels=3
        )
        
        self.assertEqual(len(memory.levels), 3)
        
        # Check that slot counts decrease by level
        for i in range(len(memory.levels) - 1):
            current_slots = memory.levels[i]['keys'].shape[0]
            next_slots = memory.levels[i + 1]['keys'].shape[0]
            self.assertGreater(current_slots, next_slots)


class TestIntegration(unittest.TestCase):
    """Test cases for integrated model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = get_device()
        self.batch_size = 2
        self.seq_len = 32
        self.vocab_size = 1000
        self.d_model = 256
        
    def test_integrated_model_creation(self):
        """Test integrated model creation."""
        model = create_integrated_model(
            vocab_size=self.vocab_size,
            model_config={'d_model': self.d_model, 'n_layers': 4},
            memory_config={'memory_slots': [256, 128]}
        )
        
        self.assertIsInstance(model, MemoryIntegratedDNC)
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)
    
    def test_integrated_model_forward(self):
        """Test integrated model forward pass."""
        model = MemoryIntegratedDNC(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=4,
            memory_slots=[256, 128]
        )
        
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        logits, metadata = model(input_ids, return_memory_stats=True)
        
        # Check outputs
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertIn('memory_stats', metadata)
        self.assertIn('write_strength', metadata)
    
    def test_integrated_model_generation(self):
        """Test integrated model generation."""
        model = MemoryIntegratedDNC(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=4,
            memory_slots=[256, 128]
        )
        
        input_ids = torch.randint(0, self.vocab_size, (1, 10))
        
        # Test generation with memory
        generated = model.generate(input_ids, max_length=20, use_memory=True)
        self.assertEqual(generated.shape, (1, 20))
        
        # Test generation without memory
        generated_no_mem = model.generate(input_ids, max_length=20, use_memory=False)
        self.assertEqual(generated_no_mem.shape, (1, 20))


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_config(self):
        """Test configuration management."""
        config_dict = {
            'model': {'d_model': 512, 'n_layers': 6},
            'training': {'lr': 0.001, 'batch_size': 32}
        }
        
        config = Config(config_dict)
        
        # Test getting values
        self.assertEqual(config.get('model.d_model'), 512)
        self.assertEqual(config.get('training.lr'), 0.001)
        self.assertEqual(config.get('nonexistent', 'default'), 'default')
        
        # Test setting values
        config.set('model.dropout', 0.1)
        self.assertEqual(config.get('model.dropout'), 0.1)
    
    def test_metrics_tracker(self):
        """Test metrics tracking."""
        tracker = MetricsTracker(window_size=5)
        
        # Update metrics
        for i in range(10):
            tracker.update({'loss': 1.0 - i * 0.1, 'accuracy': i * 0.1})
        
        # Check averages
        averages = tracker.get_averages()
        self.assertIn('loss', averages)
        self.assertIn('accuracy', averages)
        
        # Check best values
        best_values = tracker.get_best_values()
        self.assertIn('loss', best_values)
        self.assertIn('accuracy', best_values)


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.device = get_device()
    
    def benchmark_model_forward(self, model, input_shape, num_runs=10):
        """Benchmark model forward pass."""
        batch_size, seq_len = input_shape
        vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else 1000
        
        model.eval()
        model = model.to(self.device)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        tokens_per_second = (batch_size * seq_len) / avg_time
        
        return {
            'avg_time_per_forward': avg_time,
            'tokens_per_second': tokens_per_second,
            'memory_usage_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }
    
    def benchmark_memory_operations(self, memory, batch_size=4, seq_len=128, num_runs=10):
        """Benchmark memory read/write operations."""
        d_model = memory.d_model
        memory = memory.to(self.device)
        
        queries = torch.randn(batch_size, seq_len, d_model).to(self.device)
        writes = torch.randn(batch_size, seq_len, d_model).to(self.device)
        
        # Benchmark read
        start_time = time.time()
        for _ in range(num_runs):
            _ = memory.read(queries)
        read_time = (time.time() - start_time) / num_runs
        
        # Benchmark write
        start_time = time.time()
        for _ in range(num_runs):
            memory.write(writes)
        write_time = (time.time() - start_time) / num_runs
        
        return {
            'read_time': read_time,
            'write_time': write_time,
            'total_memory_ops_time': read_time + write_time
        }


def run_tests():
    """Run all tests."""
    print("Running BWR-DNC 002 Test Suite...")
    print("=" * 50)
    
    # Run unit tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestCore))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestMemory))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestIntegration))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestUtils))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


def run_benchmarks():
    """Run performance benchmarks."""
    print("\nRunning Performance Benchmarks...")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    
    # Test basic DNC model
    print("Benchmarking Basic DNC Model...")
    basic_model = DNC(vocab_size=10000, d_model=512, n_layers=6, n_heads=8)
    basic_results = benchmark.benchmark_model_forward(basic_model, (4, 128))
    
    print(f"Basic DNC - Forward pass: {basic_results['avg_time_per_forward']:.4f}s")
    print(f"Basic DNC - Tokens/sec: {basic_results['tokens_per_second']:.0f}")
    print(f"Basic DNC - Memory usage: {basic_results['memory_usage_mb']:.1f}MB")
    
    # Test integrated model
    print("\nBenchmarking Integrated DNC Model...")
    integrated_model = create_integrated_model(
        vocab_size=10000,
        model_config={'d_model': 512, 'n_layers': 6, 'n_heads': 8}
    )
    integrated_results = benchmark.benchmark_model_forward(integrated_model, (4, 128))
    
    print(f"Integrated DNC - Forward pass: {integrated_results['avg_time_per_forward']:.4f}s")
    print(f"Integrated DNC - Tokens/sec: {integrated_results['tokens_per_second']:.0f}")
    print(f"Integrated DNC - Memory usage: {integrated_results['memory_usage_mb']:.1f}MB")
    
    # Test memory operations
    print("\nBenchmarking Memory Operations...")
    memory = create_hierarchical_memory(d_model=512, base_slots=2048, levels=3)
    memory_results = benchmark.benchmark_memory_operations(memory)
    
    print(f"Memory read time: {memory_results['read_time']:.4f}s")
    print(f"Memory write time: {memory_results['write_time']:.4f}s")
    print(f"Total memory ops time: {memory_results['total_memory_ops_time']:.4f}s")
    
    print("\n" + "=" * 50)
    print("Benchmarking complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='BWR-DNC 002 Test Runner')
    parser.add_argument('--tests', action='store_true', help='Run unit tests')
    parser.add_argument('--benchmarks', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--all', action='store_true', help='Run tests and benchmarks')
    
    args = parser.parse_args()
    
    if args.all or (not args.tests and not args.benchmarks):
        # Run everything by default
        success = run_tests()
        if success:
            run_benchmarks()
        else:
            print("Tests failed, skipping benchmarks.")
    elif args.tests:
        run_tests()
    elif args.benchmarks:
        run_benchmarks()
