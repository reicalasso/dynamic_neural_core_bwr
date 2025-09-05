# BWR-DNC Mini - Modular Structure

This directory contains the modularized BWR-DNC mini implementation with organized test, debug, and documentation structure.

## Directory Structure

```
bwr-dnc_mini_000/
├── .gitignore              # Git ignore rules
├── api/                    # API server
├── backend/                # Core backend implementation
│   ├── bwr/               # BWR-DNC core modules
│   ├── train_*.py         # Training scripts
│   ├── evaluate_model.py  # Model evaluation
│   └── requirements.txt   # Backend dependencies
├── tests/                  # Test suite (modular)
│   ├── __init__.py
│   ├── test_runner.py     # Test runner configuration
│   ├── unit/              # Unit tests
│   │   ├── __init__.py
│   │   └── test_*.py      # Individual unit test files
│   ├── integration/       # Integration tests
│   │   └── __init__.py
│   └── performance/       # Performance tests
│       └── __init__.py
├── debug/                  # Debug utilities (modular)
│   ├── __init__.py
│   └── debug_*.py         # Debug scripts
├── docs/                   # Documentation (modular)
│   ├── DETAILED_PHASE_PLAN.md
│   ├── HIERARCHICAL_MEMORY_PLAN.md
│   ├── PROJECT_ROADMAP.md
│   └── *.md               # Other documentation files
└── checkpoints/           # Model checkpoints
```

## Usage

### Running Tests

```bash
# Run all tests
python tests/test_runner.py

# Run specific test categories
python tests/test_runner.py --unit
python tests/test_runner.py --integration
python tests/test_runner.py --performance

# Use pytest (if available)
python tests/test_runner.py --pytest
```

### Debug Scripts

Debug scripts are located in the `debug/` directory and can be run independently:

```bash
python debug/debug_dnc.py
python debug/debug_gradients.py
# etc.
```

### Training

Training scripts remain in the backend directory:

```bash
cd backend
python train_basic_dnc.py
python train_hierarchical_dnc.py
# etc.
```

## Benefits of Modular Structure

1. **Organized Testing**: Tests are categorized by type (unit, integration, performance)
2. **Clean Separation**: Debug utilities are separate from main code
3. **Better Documentation**: All docs are centralized in one location
4. **Easier Maintenance**: Clear structure makes it easier to find and maintain code
5. **Git Management**: Proper .gitignore prevents committing unwanted files

## Test Categories

### Unit Tests
- Test individual components in isolation
- Fast execution
- Mock external dependencies

### Integration Tests
- Test component interactions
- End-to-end workflows
- Real data processing

### Performance Tests
- Benchmarking
- Memory usage analysis
- Training speed optimization
