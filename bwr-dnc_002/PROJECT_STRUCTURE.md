# BWR-DNC 002 Project Structure

This file describes the organization of the bwr-dnc_002 project.

## Core Components

1. **Core Model** (`core/`)
   - Model architecture and implementations
   - Basic building blocks (attention, normalization, etc.)

2. **Memory System** (`memory/`)
   - State bank implementations
   - Memory management and compression
   - Eviction policies

3. **Utilities** (`utils/`)
   - Helper functions
   - Common data structures
   - Configuration management

4. **Research Tools** (`research/`)
   - Analysis and visualization tools
   - Metrics collection
   - Experimental features

5. **API** (`api/`)
   - REST API endpoints
   - WebSocket interfaces
   - Server implementation

6. **Tests** (`tests/`)
   - Unit tests for all components
   - Integration tests
   - Performance benchmarks

7. **Examples** (`examples/`)
   - Usage examples
   - Tutorials
   - Sample applications

## Design Principles

1. **Modularity**: Each component should have a single responsibility
2. **Documentation**: Every module must have clear docstrings and README files
3. **Testability**: All components should be easily testable
4. **Performance**: Optimize critical paths without sacrificing readability
5. **Simplicity**: Favor simple solutions over complex ones when possible