import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import time

@dataclass
class CurriculumStage:
    """Represents a stage in curriculum learning."""
    name: str
    tasks: List[str]
    task_weights: Dict[str, float]
    difficulty_range: Tuple[float, float]
    min_accuracy: float
    duration_steps: int
    adaptive_criteria: Dict[str, float]

class AdaptiveDifficultyManager:
    """Manages adaptive difficulty adjustment based on model performance."""
    
    def __init__(self, initial_difficulty: float = 0.3, 
                 adaptation_rate: float = 0.1,
                 performance_window: int = 100):
        self.current_difficulty = initial_difficulty
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        
        # Performance tracking
        self.performance_history = deque(maxlen=performance_window)
        self.task_difficulties = defaultdict(float)
        self.task_performance = defaultdict(lambda: deque(maxlen=50))
        
        # Adaptation parameters
        self.target_accuracy = 0.75
        self.difficulty_bounds = (0.1, 1.0)
        
    def update_performance(self, task: str, accuracy: float, loss: float):
        """Update performance metrics for a specific task."""
        performance_score = accuracy * (1.0 / (loss + 1e-8))
        
        self.performance_history.append(performance_score)
        self.task_performance[task].append(accuracy)
        
        # Adapt difficulty based on performance
        self._adapt_difficulty(task, accuracy)
    
    def _adapt_difficulty(self, task: str, accuracy: float):
        """Adapt difficulty based on current performance."""
        if accuracy > self.target_accuracy + 0.1:
            # Model is performing too well, increase difficulty
            self.task_difficulties[task] = min(
                self.difficulty_bounds[1],
                self.task_difficulties[task] + self.adaptation_rate
            )
        elif accuracy < self.target_accuracy - 0.1:
            # Model is struggling, decrease difficulty
            self.task_difficulties[task] = max(
                self.difficulty_bounds[0],
                self.task_difficulties[task] - self.adaptation_rate
            )
        
        # Update global difficulty as weighted average
        if self.task_difficulties:
            self.current_difficulty = sum(self.task_difficulties.values()) / len(self.task_difficulties)
    
    def get_task_difficulty(self, task: str) -> float:
        """Get current difficulty for a specific task."""
        return self.task_difficulties.get(task, self.current_difficulty)
    
    def should_advance_curriculum(self, current_stage: CurriculumStage) -> bool:
        """Determine if model is ready to advance to next curriculum stage."""
        if len(self.performance_history) < self.performance_window // 2:
            return False
        
        recent_performance = list(self.performance_history)[-50:]
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        return avg_performance > current_stage.min_accuracy

class CurriculumLearningManager:
    """Manages curriculum learning progression through multiple stages."""
    
    def __init__(self):
        self.stages = self._define_curriculum_stages()
        self.current_stage_idx = 0
        self.stage_start_step = 0
        self.difficulty_manager = AdaptiveDifficultyManager()
        
        # Progress tracking
        self.stage_history = []
        self.transition_log = []
    
    def _define_curriculum_stages(self) -> List[CurriculumStage]:
        """Define the curriculum learning stages."""
        return [
            CurriculumStage(
                name="Foundation",
                tasks=["copy"],
                task_weights={"copy": 1.0},
                difficulty_range=(0.1, 0.4),
                min_accuracy=0.7,
                duration_steps=1000,
                adaptive_criteria={"loss_threshold": 1.0, "accuracy_threshold": 0.7}
            ),
            CurriculumStage(
                name="Basic Reasoning",
                tasks=["copy", "lookup"],
                task_weights={"copy": 0.6, "lookup": 0.4},
                difficulty_range=(0.3, 0.6),
                min_accuracy=0.65,
                duration_steps=1500,
                adaptive_criteria={"loss_threshold": 0.8, "accuracy_threshold": 0.65}
            ),
            CurriculumStage(
                name="Intermediate",
                tasks=["copy", "lookup", "long_infill"],
                task_weights={"copy": 0.3, "lookup": 0.4, "long_infill": 0.3},
                difficulty_range=(0.4, 0.7),
                min_accuracy=0.6,
                duration_steps=2000,
                adaptive_criteria={"loss_threshold": 0.6, "accuracy_threshold": 0.6}
            ),
            CurriculumStage(
                name="Advanced",
                tasks=["copy", "lookup", "long_infill", "needle_haystack"],
                task_weights={"copy": 0.2, "lookup": 0.3, "long_infill": 0.3, "needle_haystack": 0.2},
                difficulty_range=(0.5, 0.8),
                min_accuracy=0.55,
                duration_steps=2500,
                adaptive_criteria={"loss_threshold": 0.5, "accuracy_threshold": 0.55}
            ),
            CurriculumStage(
                name="Expert",
                tasks=["copy", "lookup", "long_infill", "needle_haystack", "associative_recall"],
                task_weights={"copy": 0.1, "lookup": 0.2, "long_infill": 0.3, 
                             "needle_haystack": 0.2, "associative_recall": 0.2},
                difficulty_range=(0.6, 1.0),
                min_accuracy=0.5,
                duration_steps=3000,
                adaptive_criteria={"loss_threshold": 0.4, "accuracy_threshold": 0.5}
            )
        ]
    
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    def should_advance(self, current_step: int) -> bool:
        """Check if curriculum should advance to next stage."""
        current_stage = self.get_current_stage()
        
        # Check minimum duration
        steps_in_stage = current_step - self.stage_start_step
        if steps_in_stage < current_stage.duration_steps:
            return False
        
        # Check performance criteria
        return self.difficulty_manager.should_advance_curriculum(current_stage)
    
    def advance_stage(self, current_step: int) -> bool:
        """Advance to the next curriculum stage."""
        if self.current_stage_idx < len(self.stages) - 1:
            old_stage = self.get_current_stage()
            self.current_stage_idx += 1
            new_stage = self.get_current_stage()
            
            self.stage_start_step = current_step
            
            # Log transition
            self.transition_log.append({
                'step': current_step,
                'from_stage': old_stage.name,
                'to_stage': new_stage.name,
                'reason': 'performance_criteria_met'
            })
            
            return True
        return False
    
    def sample_task(self) -> str:
        """Sample a task from current stage based on weights."""
        current_stage = self.get_current_stage()
        tasks = list(current_stage.task_weights.keys())
        weights = list(current_stage.task_weights.values())
        
        return np.random.choice(tasks, p=weights)
    
    def get_difficulty_for_task(self, task: str) -> float:
        """Get current difficulty for a task."""
        current_stage = self.get_current_stage()
        base_difficulty = self.difficulty_manager.get_task_difficulty(task)
        
        # Clamp to stage range
        min_diff, max_diff = current_stage.difficulty_range
        return np.clip(base_difficulty, min_diff, max_diff)


class SmartBatchSampler:
    """Intelligent batch sampling for improved training efficiency."""
    
    def __init__(self, dataset, batch_size: int = 32, 
                 difficulty_balancing: bool = True,
                 sequence_length_balancing: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.difficulty_balancing = difficulty_balancing
        self.sequence_length_balancing = sequence_length_balancing
        
        # Sample categorization
        self.samples_by_difficulty = defaultdict(list)
        self.samples_by_length = defaultdict(list)
        self._categorize_samples()
    
    def _categorize_samples(self):
        """Categorize samples by difficulty and length."""
        for idx, sample in enumerate(self.dataset):
            # Estimate difficulty (this would be more sophisticated in practice)
            difficulty = self._estimate_difficulty(sample)
            length = len(sample[0]) if hasattr(sample[0], '__len__') else 128
            
            self.samples_by_difficulty[difficulty].append(idx)
            self.samples_by_length[length // 32].append(idx)  # Group by length buckets
    
    def _estimate_difficulty(self, sample) -> str:
        """Estimate the difficulty of a sample."""
        # Simple heuristic - would be more sophisticated in practice
        if hasattr(sample, '__len__') and len(sample) > 2:
            task_name = sample[2] if len(sample) > 2 else 'unknown'
            return task_name
        return 'medium'
    
    def sample_balanced_batch(self, curriculum_manager: CurriculumLearningManager) -> List[int]:
        """Sample a balanced batch based on current curriculum."""
        current_stage = curriculum_manager.get_current_stage()
        batch_indices = []
        
        # Sample according to task weights
        for task, weight in current_stage.task_weights.items():
            task_samples = int(self.batch_size * weight)
            if task in self.samples_by_difficulty:
                available_samples = self.samples_by_difficulty[task]
                if available_samples:
                    selected = random.sample(
                        available_samples, 
                        min(task_samples, len(available_samples))
                    )
                    batch_indices.extend(selected)
        
        # Fill remaining slots randomly
        while len(batch_indices) < self.batch_size:
            all_samples = list(range(len(self.dataset)))
            remaining = [i for i in all_samples if i not in batch_indices]
            if remaining:
                batch_indices.append(random.choice(remaining))
            else:
                break
        
        return batch_indices[:self.batch_size]


class AdvancedOptimizationScheduler:
    """Advanced learning rate and optimization scheduling."""
    
    def __init__(self, optimizer, warmup_steps: int = 1000, 
                 max_lr: float = 1e-3, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
        
        # Adaptive scheduling
        self.performance_history = deque(maxlen=100)
        self.lr_adaptation_enabled = True
        self.plateau_patience = 50
        self.plateau_factor = 0.8
        self.steps_since_improvement = 0
        self.best_performance = float('-inf')
    
    def step(self, performance_metric: Optional[float] = None):
        """Update learning rate based on schedule and performance."""
        self.current_step += 1
        
        # Warmup phase
        if self.current_step <= self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing with restarts
            cycle_length = 5000
            cycle_progress = (self.current_step - self.warmup_steps) % cycle_length
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * cycle_progress / cycle_length)
            )
        
        # Adaptive adjustment based on performance
        if performance_metric is not None and self.lr_adaptation_enabled:
            self.performance_history.append(performance_metric)
            
            if performance_metric > self.best_performance:
                self.best_performance = performance_metric
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1
            
            # Plateau detection
            if (self.steps_since_improvement > self.plateau_patience and 
                len(self.performance_history) >= self.plateau_patience):
                recent_performance = list(self.performance_history)[-self.plateau_patience:]
                if max(recent_performance) - min(recent_performance) < 0.01:
                    # Plateau detected, reduce learning rate
                    lr *= self.plateau_factor
                    self.steps_since_improvement = 0
        
        # Apply learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class ModelCompressionOptimizer:
    """Optimize model through pruning and quantization during training."""
    
    def __init__(self, model, compression_ratio: float = 0.1):
        self.model = model
        self.compression_ratio = compression_ratio
        self.pruning_enabled = True
        self.quantization_enabled = True
        
        # Tracking
        self.original_size = self._calculate_model_size()
        self.compression_history = []
    
    def _calculate_model_size(self) -> int:
        """Calculate total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    def structured_pruning(self, importance_threshold: float = 0.1):
        """Perform structured pruning based on parameter importance."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate importance scores
                if hasattr(module, 'weight'):
                    weight_importance = torch.norm(module.weight, dim=1)
                    threshold = torch.quantile(weight_importance, self.compression_ratio)
                    
                    # Create mask for important weights
                    mask = weight_importance > threshold
                    
                    # Apply mask (this is a simplified version)
                    with torch.no_grad():
                        module.weight.data[~mask] = 0
    
    def knowledge_distillation_loss(self, student_outputs, teacher_outputs, 
                                  temperature: float = 3.0, alpha: float = 0.5):
        """Compute knowledge distillation loss."""
        soft_student = F.log_softmax(student_outputs / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_outputs / temperature, dim=-1)
        
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        return alpha * kd_loss * (temperature ** 2)


class AdvancedTrainingManager:
    """Comprehensive training manager with all advanced features."""
    
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Initialize components
        self.curriculum_manager = CurriculumLearningManager()
        self.batch_sampler = None  # Will be set when dataset is available
        self.lr_scheduler = None   # Will be set when optimizer is available
        self.compression_optimizer = ModelCompressionOptimizer(model)
        
        # Training state
        self.current_step = 0
        self.metrics_history = []
        
        # Experiment tracking
        self.experiment_log = {
            'curriculum_transitions': [],
            'performance_milestones': [],
            'optimization_events': []
        }
    
    def setup_training(self, dataset, optimizer):
        """Setup training components."""
        batch_size = self.config.get('batch_size', 32)
        self.batch_sampler = SmartBatchSampler(dataset, batch_size)
        
        warmup_steps = self.config.get('warmup_steps', 1000)
        max_lr = self.config.get('max_lr', 1e-3)
        self.lr_scheduler = AdvancedOptimizationScheduler(
            optimizer, warmup_steps, max_lr
        )
    
    def training_step(self, batch) -> Dict[str, Any]:
        """Execute one training step with all optimizations."""
        # Get current task and difficulty
        current_task = self.curriculum_manager.sample_task()
        difficulty = self.curriculum_manager.get_difficulty_for_task(current_task)
        
        # Forward pass
        inputs, targets = batch[0], batch[1]
        logits, aux_info = self.model(inputs, return_state_info=True)
        
        # Compute loss
        main_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        # Add auxiliary losses
        total_loss = main_loss
        
        # Compute metrics
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        # Update curriculum and difficulty
        self.curriculum_manager.difficulty_manager.update_performance(
            current_task, accuracy.item(), main_loss.item()
        )
        
        # Check curriculum advancement
        if self.curriculum_manager.should_advance(self.current_step):
            if self.curriculum_manager.advance_stage(self.current_step):
                self.experiment_log['curriculum_transitions'].append({
                    'step': self.current_step,
                    'new_stage': self.curriculum_manager.get_current_stage().name
                })
        
        # Update learning rate
        current_lr = self.lr_scheduler.step(accuracy.item())
        
        # Periodic model compression
        if (self.current_step % 1000 == 0 and 
            self.config.get('enable_compression', False)):
            self.compression_optimizer.structured_pruning()
        
        self.current_step += 1
        
        return {
            'loss': total_loss,
            'accuracy': accuracy,
            'learning_rate': current_lr,
            'current_task': current_task,
            'difficulty': difficulty,
            'curriculum_stage': self.curriculum_manager.get_current_stage().name,
            'aux_info': aux_info
        }
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        current_stage = self.curriculum_manager.get_current_stage()
        
        stats = {
            'current_step': self.current_step,
            'curriculum_stage': current_stage.name,
            'stage_progress': (self.current_step - self.curriculum_manager.stage_start_step) / current_stage.duration_steps,
            'task_difficulties': dict(self.curriculum_manager.difficulty_manager.task_difficulties),
            'model_size': self.compression_optimizer._calculate_model_size(),
            'compression_ratio': 1.0 - (self.compression_optimizer._calculate_model_size() / self.compression_optimizer.original_size),
            'experiment_log': self.experiment_log
        }
        
        return stats
