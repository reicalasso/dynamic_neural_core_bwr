"""
BWR-DNC 002: Utility Functions

This module provides common utility functions and data structures
used throughout the BWR-DNC project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import yaml
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging


class Config:
    """
    Configuration management class.
    
    Provides a clean interface for loading and managing configuration
    parameters from various sources (dict, JSON, YAML).
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Initial configuration dictionary
        """
        self._config = config_dict or {}
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Config instance
        """
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation like 'model.d_model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def save(self, config_path: Union[str, Path]):
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


class Logger:
    """
    Simple logging utility.
    
    Provides consistent logging across the project with different levels
    and optional file output.
    """
    
    def __init__(self, name: str, level: str = 'INFO', log_file: str = None):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file to write logs to
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
    
    def critical(self, msg: str):
        """Log critical message."""
        self.logger.critical(msg)


class MovingAverage:
    """
    Simple moving average for metrics tracking.
    
    Efficiently tracks moving averages of scalar values with
    configurable window size.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize moving average.
        
        Args:
            window_size: Size of the moving window
        """
        self.window_size = window_size
        self.values = []
        self.sum = 0.0
    
    def update(self, value: float):
        """
        Update moving average with new value.
        
        Args:
            value: New value to add
        """
        self.values.append(value)
        self.sum += value
        
        if len(self.values) > self.window_size:
            old_value = self.values.pop(0)
            self.sum -= old_value
    
    def avg(self) -> float:
        """Get current average."""
        if not self.values:
            return 0.0
        return self.sum / len(self.values)
    
    def reset(self):
        """Reset the moving average."""
        self.values.clear()
        self.sum = 0.0


class MetricsTracker:
    """
    Comprehensive metrics tracking system.
    
    Tracks multiple metrics with moving averages, best values,
    and history for visualization.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.metrics = {}
        self.history = {}
        self.best_values = {}
    
    def update(self, metrics: Dict[str, float]):
        """
        Update multiple metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = MovingAverage(self.window_size)
                self.history[name] = []
                self.best_values[name] = {'value': value, 'step': 0}
            
            self.metrics[name].update(value)
            self.history[name].append(value)
            
            # Update best value (assume lower is better for loss, higher for accuracy)
            if 'loss' in name.lower() or 'error' in name.lower():
                if value < self.best_values[name]['value']:
                    self.best_values[name] = {'value': value, 'step': len(self.history[name]) - 1}
            else:
                if value > self.best_values[name]['value']:
                    self.best_values[name] = {'value': value, 'step': len(self.history[name]) - 1}
    
    def get_averages(self) -> Dict[str, float]:
        """Get current moving averages for all metrics."""
        return {name: metric.avg() for name, metric in self.metrics.items()}
    
    def get_best_values(self) -> Dict[str, Dict[str, float]]:
        """Get best values for all metrics."""
        return self.best_values.copy()
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get full history for a specific metric."""
        return self.history.get(metric_name, []).copy()
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
        self.history.clear()
        self.best_values.clear()


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    total_params, _ = count_parameters(model)
    # Assume 4 bytes per parameter (float32)
    size_mb = (total_params * 4) / (1024 * 1024)
    return size_mb


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Set deterministic operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def memory_usage_mb() -> float:
    """
    Get current GPU memory usage in MB.
    
    Returns:
        Memory usage in MB (0 if no GPU)
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: Union[str, Path],
    metadata: Dict[str, Any] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        checkpoint_path: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint metadata
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metadata': checkpoint.get('metadata', {})
    }


class EarlyStopping:
    """
    Early stopping utility.
    
    Monitors a metric and stops training when it stops improving
    for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric_value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False
        
        if self.mode == 'min':
            improved = metric_value < self.best_value - self.min_delta
        else:
            improved = metric_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def calculate_model_params(model: nn.Module) -> int:
    """
    Calculate the total number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def format_memory_size(num_bytes: Union[int, float]) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "256 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            if unit == 'B':
                return f"{int(num_bytes)} {unit}"
            else:
                return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
