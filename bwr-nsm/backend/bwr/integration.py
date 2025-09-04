"""
BWR-NSM Integration Module
==========================

This module integrates all the advanced features of BWR-NSM:
- State Persistence
- Asynchronous Processing  
- Performance Monitoring
- Advanced Eviction Policies
- Unlimited Context Support
- Advanced Training Features

Usage:
    from bwr.integration import create_advanced_nsm, AdvancedNSMTrainer
    
    # Create model with all features
    model = create_advanced_nsm(config)
    
    # Create trainer with all optimizations
    trainer = AdvancedNSMTrainer(model, config)
    trainer.train()
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

# Import all components
from .model import NSM
from .statebank import AdvancedStateBank
from .persistence import StatePersistenceManager
from .async_manager import AsyncMemoryManager, AsyncTrainingManager
from .performance_monitor import RTXPerformanceMonitor, ModelPerformanceAnalyzer
from .eviction_policies import AdvancedEvictionManager, SmartMemoryManager, ContextAwareEviction
from .unlimited_context import UnlimitedContextNSM, DynamicCompressionManager
from .advanced_training import AdvancedTrainingManager, CurriculumLearningManager

logger = logging.getLogger(__name__)

class AdvancedNSM(nn.Module):
    """Enhanced NSM with all advanced features integrated."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Initialize base model
        model_config = config['model']
        self.base_model = NSM(
            vocab=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            slots=model_config['slots'],
            max_seq_len=model_config.get('max_seq_len', 2048),
            dropout=model_config.get('dropout', 0.1)
        )
        
        # Enhance with unlimited context capability
        if config.get('unlimited_context', True):
            self.model = UnlimitedContextNSM(
                self.base_model, 
                base_context_length=model_config.get('max_seq_len', 512)
            )
        else:
            self.model = self.base_model
        
        # Initialize advanced state bank with eviction policies
        eviction_manager = AdvancedEvictionManager()
        smart_memory = SmartMemoryManager(eviction_manager)
        context_eviction = ContextAwareEviction()
        
        # Replace state bank with advanced version
        self.model.state = AdvancedStateBank(
            model_config['d_model'],
            slots=model_config['slots'],
            heads=model_config['n_heads']
        )
        
        # Performance monitoring
        self.performance_monitor = RTXPerformanceMonitor()
        self.model_analyzer = ModelPerformanceAnalyzer()
        
        # Async memory management
        self.async_manager = AsyncMemoryManager(self.model.state)
        
        # State persistence
        self.persistence_manager = StatePersistenceManager()
        
        # Training enhancements
        self.training_manager = AdvancedTrainingManager(self.model, config)
        
        logger.info("Advanced NSM initialized with all features")
    
    def forward(self, input_ids, **kwargs):
        """Forward pass with enhanced features."""
        return self.model(input_ids, **kwargs)
    
    async def async_forward(self, input_ids, **kwargs):
        """Asynchronous forward pass with background memory updates."""
        # Regular forward pass
        output, aux_info = self.forward(input_ids, **kwargs)
        
        # Queue async memory updates
        if aux_info and 'hidden_states' in aux_info:
            await self.async_manager.queue_memory_update(
                data=aux_info['hidden_states'].mean(dim=1),
                priority=1.0,
                metadata={'timestamp': torch.now().timestamp()}
            )
        
        return output, aux_info
    
    def save_state(self, session_id: Optional[str] = None, global_save: bool = False):
        """Save current model state with persistence."""
        return self.persistence_manager.save_state_bank(
            self.model.state, session_id, global_save
        )
    
    def load_state(self, session_id: Optional[str] = None, latest_global: bool = False):
        """Load model state from persistence."""
        return self.persistence_manager.load_state_bank(
            self.model.state, session_id, latest_global
        )
    
    def get_performance_metrics(self):
        """Get current performance metrics."""
        metrics = self.performance_monitor.get_real_time_metrics()
        
        # Add model-specific metrics
        if hasattr(self.model, 'state'):
            state_util = self.model_analyzer.analyze_state_bank_utilization(self.model.state)
            compression_ratio = self.model_analyzer.calculate_compression_ratio(self.model.state)
            
            metrics.update({
                'state_bank_utilization': state_util,
                'compression_ratio': compression_ratio
            })
        
        return metrics
    
    def get_compression_stats(self):
        """Get compression statistics if unlimited context is enabled."""
        if isinstance(self.model, UnlimitedContextNSM):
            return self.model.get_compression_statistics()
        return {}
    
    async def start_monitoring(self):
        """Start all background monitoring and async processes."""
        self.performance_monitor.start_monitoring()
        await self.async_manager.start()
        logger.info("All monitoring systems started")
    
    async def stop_monitoring(self):
        """Stop all background processes."""
        self.performance_monitor.stop_monitoring()
        await self.async_manager.stop()
        logger.info("All monitoring systems stopped")


class AdvancedNSMTrainer:
    """Comprehensive trainer with all advanced features."""
    
    def __init__(self, model: AdvancedNSM, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        # Setup training manager
        self.training_manager = model.training_manager
        
        # Performance tracking
        self.metrics_history = []
        self.best_performance = 0.0
        
        logger.info("Advanced NSM Trainer initialized")
    
    async def train(self, dataset, epochs: int = None):
        """Train with all advanced features."""
        epochs = epochs or self.config['training']['epochs']
        
        # Setup training components
        self.training_manager.setup_training(dataset, self.optimizer)
        
        # Start monitoring
        await self.model.start_monitoring()
        
        try:
            for epoch in range(epochs):
                await self._train_epoch(dataset, epoch)
                
                # Evaluate periodically
                if epoch % 5 == 0:
                    metrics = await self._evaluate()
                    logger.info(f"Epoch {epoch}: {metrics}")
                    
                    # Save checkpoint if performance improved
                    if metrics.get('accuracy', 0) > self.best_performance:
                        self.best_performance = metrics['accuracy']
                        self.model.save_state(global_save=True)
        
        finally:
            await self.model.stop_monitoring()
    
    async def _train_epoch(self, dataset, epoch: int):
        """Train one epoch with advanced features."""
        self.model.train()
        
        # Use smart batch sampling
        batch_sampler = self.training_manager.batch_sampler
        
        for step in range(len(dataset) // self.config['training']['batch_size']):
            # Sample intelligent batch
            batch_indices = batch_sampler.sample_balanced_batch(
                self.training_manager.curriculum_manager
            )
            batch = [dataset[i] for i in batch_indices]
            
            # Convert to tensors
            inputs = torch.stack([torch.tensor(item[0]) for item in batch])
            targets = torch.stack([torch.tensor(item[1]) for item in batch])
            
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            
            # Training step with all optimizations
            step_metrics = self.training_manager.training_step((inputs, targets))
            
            # Backward pass
            self.optimizer.zero_grad()
            step_metrics['loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training'].get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Record performance metrics
            if self.model.performance_monitor:
                self.model.performance_monitor.record_training_step(
                    batch_size=len(batch),
                    forward_time=0.01,  # Would measure actual time
                    backward_time=0.01,
                    loss=step_metrics['loss'].item(),
                    lr=step_metrics['learning_rate']
                )
            
            # Log progress
            if step % 100 == 0:
                stats = self.training_manager.get_training_statistics()
                logger.info(f"Epoch {epoch}, Step {step}: {stats}")
    
    async def _evaluate(self) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        
        with torch.no_grad():
            # Simple evaluation (would be more comprehensive in practice)
            total_accuracy = 0.0
            num_batches = 0
            
            # Mock evaluation
            for _ in range(10):  # 10 evaluation batches
                mock_inputs = torch.randint(0, 1000, (4, 128))
                mock_targets = torch.randint(0, 1000, (4, 128))
                
                if torch.cuda.is_available():
                    mock_inputs = mock_inputs.cuda()
                    mock_targets = mock_targets.cuda()
                
                outputs, _ = self.model(mock_inputs)
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == mock_targets).float().mean()
                
                total_accuracy += accuracy.item()
                num_batches += 1
        
        return {
            'accuracy': total_accuracy / num_batches,
            'performance_metrics': self.model.get_performance_metrics(),
            'compression_stats': self.model.get_compression_stats()
        }


def create_advanced_nsm(config: Dict[str, Any]) -> AdvancedNSM:
    """Factory function to create an Advanced NSM with all features."""
    return AdvancedNSM(config)


def create_trainer(model: AdvancedNSM, config: Dict[str, Any]) -> AdvancedNSMTrainer:
    """Factory function to create an Advanced NSM Trainer."""
    return AdvancedNSMTrainer(model, config)


# Example usage
def example_usage():
    """Example of how to use the advanced features."""
    
    # Configuration
    config = {
        'model': {
            'vocab_size': 32000,
            'd_model': 512,
            'n_layers': 8,
            'n_heads': 8,
            'slots': 2048,
            'max_seq_len': 1024,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 8,
            'lr': 1e-4,
            'weight_decay': 0.01,
            'epochs': 20,
            'grad_clip': 1.0
        },
        'unlimited_context': True,
        'enable_compression': True
    }
    
    # Create model and trainer
    model = create_advanced_nsm(config)
    trainer = create_trainer(model, config)
    
    # Mock dataset
    dataset = [(torch.randint(0, 1000, (128,)), torch.randint(0, 1000, (128,))) for _ in range(1000)]
    
    # Train (in async context)
    # await trainer.train(dataset)
    
    return model, trainer

if __name__ == "__main__":
    # Example usage
    model, trainer = example_usage()
    print("Advanced NSM system ready!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Features enabled: {list(model.config.keys())}")
