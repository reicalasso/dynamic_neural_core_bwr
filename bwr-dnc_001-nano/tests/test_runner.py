"""
Test configuration and runner for BWR-DNC mini implementation.
"""

import os
import sys
import unittest
import pytest
from pathlib import Path

# Add the backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

def run_unit_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "unit"
    suite = loader.discover(str(start_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

def run_integration_tests():
    """Run all integration tests."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "integration"
    suite = loader.discover(str(start_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

def run_performance_tests():
    """Run all performance tests."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "performance"
    suite = loader.discover(str(start_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

def run_all_tests():
    """Run all tests."""
    print("Running Unit Tests...")
    unit_result = run_unit_tests()
    
    print("\nRunning Integration Tests...")
    integration_result = run_integration_tests()
    
    print("\nRunning Performance Tests...")
    performance_result = run_performance_tests()
    
    return unit_result, integration_result, performance_result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BWR-DNC Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--pytest", action="store_true", help="Use pytest instead of unittest")
    
    args = parser.parse_args()
    
    if args.pytest:
        # Use pytest if specified
        test_dir = Path(__file__).parent
        if args.unit:
            pytest.main([str(test_dir / "unit"), "-v"])
        elif args.integration:
            pytest.main([str(test_dir / "integration"), "-v"])
        elif args.performance:
            pytest.main([str(test_dir / "performance"), "-v"])
        else:
            pytest.main([str(test_dir), "-v"])
    else:
        # Use unittest
        if args.unit:
            run_unit_tests()
        elif args.integration:
            run_integration_tests()
        elif args.performance:
            run_performance_tests()
        else:
            run_all_tests()
