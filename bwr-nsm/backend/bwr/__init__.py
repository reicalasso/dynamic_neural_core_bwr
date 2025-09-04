"""
BWR-NSM: A Neural State Machine Implementation

This package implements a Neural State Machine (NSM) with hierarchical memory management
and adaptive compression for efficient long-range context modeling.

Components:
- NSM: The main neural state machine model
- StateBank: Hierarchical memory management system
- Trainer: Training utilities and optimization strategies
- Dataset: Data loading and preprocessing utilities
- Utils: Helper functions and utilities

Advanced Features (v2.0):
- State Persistence: Cross-session memory and user profiling
- Async Processing: Background memory updates and consolidation
- Performance Monitoring: Real-time GPU and system metrics
- Advanced Eviction Policies: Smart memory management strategies
- Unlimited Context: Dynamic compression for infinite context length
- Advanced Training: Curriculum learning and adaptive optimization
"""

__version__ = "2.0.0"
__author__ = "BWR Team"
__email__ = "bwr@example.com"

# Core components
from .model import NSM
from .statebank import StateBank, AdvancedStateBank
from .trainer import AdvancedTrainer
from .dataset import Dataset
from .utils import *

# Advanced features (v2.0)
from .persistence import StatePersistenceManager
from .async_manager import AsyncMemoryManager, AsyncTrainingManager
from .performance_monitor import RTXPerformanceMonitor, ModelPerformanceAnalyzer
from .eviction_policies import AdvancedEvictionManager, SmartMemoryManager, ContextAwareEviction
from .unlimited_context import UnlimitedContextNSM, DynamicCompressionManager
from .advanced_training import AdvancedTrainingManager, CurriculumLearningManager

# Integration
from .integration import AdvancedNSM, AdvancedNSMTrainer, create_advanced_nsm, create_trainer

__all__ = [
    # Core components
    "NSM",
    "StateBank", 
    "AdvancedStateBank",
    "AdvancedTrainer",
    "Dataset",
    
    # Advanced features
    "StatePersistenceManager",
    "AsyncMemoryManager",
    "AsyncTrainingManager", 
    "RTXPerformanceMonitor",
    "ModelPerformanceAnalyzer",
    "AdvancedEvictionManager",
    "SmartMemoryManager",
    "ContextAwareEviction",
    "UnlimitedContextNSM",
    "DynamicCompressionManager",
    "AdvancedTrainingManager",
    "CurriculumLearningManager",
    
    # Integration
    "AdvancedNSM",
    "AdvancedNSMTrainer",
    "create_advanced_nsm",
    "create_trainer"
]