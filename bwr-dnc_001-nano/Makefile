# BWR-DNC Mini - Makefile
# Simplified project management commands

.PHONY: help install test test-unit test-integration test-performance debug clean setup

# Default target
help:
	@echo "BWR-DNC Mini - Available Commands:"
	@echo "=================================="
	@echo "setup          - Setup development environment"
	@echo "install        - Install dependencies"
	@echo "test           - Run all tests"
	@echo "test-unit      - Run unit tests only"
	@echo "test-integration - Run integration tests only"
	@echo "test-performance - Run performance tests only"
	@echo "debug          - Open debug manager"
	@echo "clean          - Clean temporary files"
	@echo "train-basic    - Train basic DNC model"
	@echo "train-hierarchical - Train hierarchical DNC model"
	@echo "evaluate       - Evaluate trained model"

# Setup development environment
setup:
	@echo "Setting up BWR-DNC Mini development environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r backend/requirements.txt
	pip install -r api/requirements.txt
	@echo "Dependencies installed."

# Test commands
test:
	@echo "Running all tests..."
	python tests/test_runner.py

test-unit:
	@echo "Running unit tests..."
	python tests/test_runner.py --unit

test-integration:
	@echo "Running integration tests..."
	python tests/test_runner.py --integration

test-performance:
	@echo "Running performance tests..."
	python tests/test_runner.py --performance

# Debug commands
debug:
	@echo "Opening debug manager..."
	python debug/debug_manager.py

# Training commands
train-basic:
	@echo "Training basic DNC model..."
	cd backend && python train_basic_dnc.py

train-hierarchical:
	@echo "Training hierarchical DNC model..."
	cd backend && python train_hierarchical_dnc.py

# Evaluation
evaluate:
	@echo "Evaluating model..."
	cd backend && python evaluate_model.py

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "Cleanup complete."

# Development helpers
lint:
	@echo "Running code linting..."
	python -m flake8 backend/ tests/ debug/ --max-line-length=120
	python -m black --check backend/ tests/ debug/

format:
	@echo "Formatting code..."
	python -m black backend/ tests/ debug/
	python -m isort backend/ tests/ debug/

# Project structure
structure:
	@echo "Project structure:"
	@tree -I '__pycache__|*.pyc|venv|checkpoints' .
