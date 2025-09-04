import torch
import math
import time
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from enum import Enum

class EvictionPolicy(Enum):
    """Available eviction policies for memory management."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    SALIENCE = "salience"  # Pure salience-based
    LRU_SALIENCE = "lru_salience"  # Hybrid LRU + Salience
    ADAPTIVE = "adaptive"  # Adaptive policy based on context

class AdvancedEvictionManager:
    """Advanced eviction policy manager with multiple strategies."""
    
    def __init__(self, policy: EvictionPolicy = EvictionPolicy.LRU_SALIENCE, 
                 adaptive_threshold: float = 0.8):
        self.policy = policy
        self.adaptive_threshold = adaptive_threshold
        
        # Policy-specific parameters
        self.lru_weight = 0.6
        self.salience_weight = 0.4
        
        # Adaptive policy learning
        self.policy_performance = defaultdict(list)
        self.current_adaptive_policy = EvictionPolicy.LRU_SALIENCE
        
        # Statistics tracking
        self.eviction_stats = {
            'total_evictions': 0,
            'policy_switches': 0,
            'accuracy_improvements': 0
        }
    
    def select_eviction_candidate(self, level: Dict, context: Optional[Dict] = None) -> int:
        """Select the best candidate for eviction based on current policy."""
        if self.policy == EvictionPolicy.ADAPTIVE:
            return self._adaptive_eviction(level, context)
        elif self.policy == EvictionPolicy.LRU:
            return self._lru_eviction(level)
        elif self.policy == EvictionPolicy.LFU:
            return self._lfu_eviction(level)
        elif self.policy == EvictionPolicy.SALIENCE:
            return self._salience_eviction(level)
        elif self.policy == EvictionPolicy.LRU_SALIENCE:
            return self._lru_salience_eviction(level)
        else:
            return self._lru_salience_eviction(level)  # Default fallback
    
    def _lru_eviction(self, level: Dict) -> int:
        """Least Recently Used eviction."""
        ages = level['age']
        return torch.argmax(ages).item()
    
    def _lfu_eviction(self, level: Dict) -> int:
        """Least Frequently Used eviction."""
        access_counts = level['access_count']
        return torch.argmin(access_counts).item()
    
    def _salience_eviction(self, level: Dict) -> int:
        """Pure salience-based eviction."""
        salience = level['salience']
        return torch.argmin(salience).item()
    
    def _lru_salience_eviction(self, level: Dict) -> int:
        """Hybrid LRU + Salience eviction policy."""
        ages = level['age'].float()
        salience = level['salience'].float()
        
        # Normalize scores to [0, 1]
        age_normalized = ages / (ages.max() + 1e-8)
        salience_inverted = 1.0 - (salience / (salience.max() + 1e-8))
        
        # Combine scores with weights
        eviction_score = (self.lru_weight * age_normalized + 
                         self.salience_weight * salience_inverted)
        
        # Add small randomness to break ties
        noise = torch.rand_like(eviction_score) * 0.05
        final_score = eviction_score + noise
        
        return torch.argmax(final_score).item()
    
    def _adaptive_eviction(self, level: Dict, context: Optional[Dict] = None) -> int:
        """Adaptive eviction that switches between policies based on performance."""
        # Try different policies and track their effectiveness
        current_time = time.time()
        
        # Switch policy based on recent performance
        if len(self.policy_performance[self.current_adaptive_policy]) > 50:
            recent_performance = self.policy_performance[self.current_adaptive_policy][-50:]
            avg_performance = sum(recent_performance) / len(recent_performance)
            
            if avg_performance < self.adaptive_threshold:
                # Try a different policy
                policies = [EvictionPolicy.LRU, EvictionPolicy.LFU, 
                           EvictionPolicy.SALIENCE, EvictionPolicy.LRU_SALIENCE]
                
                # Remove current policy from candidates
                if self.current_adaptive_policy in policies:
                    policies.remove(self.current_adaptive_policy)
                
                if policies:
                    # Choose policy with best recent performance
                    best_policy = self.current_adaptive_policy
                    best_performance = avg_performance
                    
                    for policy in policies:
                        if len(self.policy_performance[policy]) > 10:
                            policy_perf = sum(self.policy_performance[policy][-10:]) / 10
                            if policy_perf > best_performance:
                                best_performance = policy_perf
                                best_policy = policy
                    
                    if best_policy != self.current_adaptive_policy:
                        self.current_adaptive_policy = best_policy
                        self.eviction_stats['policy_switches'] += 1
        
        # Execute eviction with current adaptive policy
        if self.current_adaptive_policy == EvictionPolicy.LRU:
            return self._lru_eviction(level)
        elif self.current_adaptive_policy == EvictionPolicy.LFU:
            return self._lfu_eviction(level)
        elif self.current_adaptive_policy == EvictionPolicy.SALIENCE:
            return self._salience_eviction(level)
        else:  # LRU_SALIENCE
            return self._lru_salience_eviction(level)
    
    def update_performance_feedback(self, policy: EvictionPolicy, 
                                  performance_metric: float):
        """Update performance feedback for adaptive learning."""
        self.policy_performance[policy].append(performance_metric)
        
        # Keep only recent history
        if len(self.policy_performance[policy]) > 1000:
            self.policy_performance[policy] = self.policy_performance[policy][-500:]
    
    def get_eviction_statistics(self) -> Dict:
        """Get eviction policy statistics."""
        stats = self.eviction_stats.copy()
        
        # Add policy performance summaries
        performance_summary = {}
        for policy, performances in self.policy_performance.items():
            if performances:
                performance_summary[policy.value] = {
                    'count': len(performances),
                    'avg_performance': sum(performances) / len(performances),
                    'recent_performance': sum(performances[-10:]) / min(10, len(performances))
                }
        
        stats['policy_performance'] = performance_summary
        stats['current_adaptive_policy'] = self.current_adaptive_policy.value
        
        return stats


class SmartMemoryManager:
    """Smart memory manager with context-aware eviction."""
    
    def __init__(self, eviction_manager: AdvancedEvictionManager):
        self.eviction_manager = eviction_manager
        
        # Context tracking
        self.access_patterns = defaultdict(int)
        self.temporal_patterns = defaultdict(list)
        self.semantic_clusters = {}
        
        # Performance tracking
        self.hit_rates = defaultdict(float)
        self.miss_penalties = defaultdict(float)
    
    def should_evict(self, level: Dict, memory_pressure: float, 
                    context: Optional[Dict] = None) -> bool:
        """Determine if eviction should occur based on memory pressure."""
        # Basic memory pressure threshold
        if memory_pressure > 0.9:
            return True
        
        # Context-aware eviction decisions
        if context:
            task_complexity = context.get('task_complexity', 0.5)
            sequence_length = context.get('sequence_length', 0)
            
            # Higher thresholds for complex tasks
            threshold = 0.8 + (task_complexity * 0.15)
            
            # Adjust based on sequence length
            if sequence_length > 1000:
                threshold -= 0.1
            
            return memory_pressure > threshold
        
        return memory_pressure > 0.85
    
    def predict_future_access(self, slot_idx: int, level_idx: int) -> float:
        """Predict likelihood of future access to a memory slot."""
        # Temporal pattern analysis
        access_key = f"{level_idx}_{slot_idx}"
        recent_accesses = self.temporal_patterns[access_key]
        
        if len(recent_accesses) < 2:
            return 0.5  # Neutral prediction
        
        # Calculate access frequency
        current_time = time.time()
        recent_window = [t for t in recent_accesses if current_time - t < 300]  # 5 minutes
        
        if not recent_window:
            return 0.1  # Low likelihood
        
        # Calculate access rate
        time_span = current_time - min(recent_window)
        access_rate = len(recent_window) / (time_span + 1e-8)
        
        # Normalize to [0, 1]
        return min(1.0, access_rate * 10)
    
    def update_access_pattern(self, slot_idx: int, level_idx: int, 
                            query_context: Optional[Dict] = None):
        """Update access patterns for future predictions."""
        access_key = f"{level_idx}_{slot_idx}"
        current_time = time.time()
        
        # Update temporal patterns
        self.temporal_patterns[access_key].append(current_time)
        
        # Keep only recent history
        if len(self.temporal_patterns[access_key]) > 100:
            self.temporal_patterns[access_key] = self.temporal_patterns[access_key][-50:]
        
        # Update frequency counter
        self.access_patterns[access_key] += 1
    
    def analyze_memory_efficiency(self, level: Dict, level_idx: int) -> Dict:
        """Analyze memory efficiency and suggest optimizations."""
        analysis = {}
        
        # Utilization analysis
        active_slots = (level['salience'] > 0.1).sum().item()
        total_slots = level['salience'].numel()
        utilization = active_slots / total_slots
        
        analysis['utilization'] = utilization
        analysis['active_slots'] = active_slots
        analysis['total_slots'] = total_slots
        
        # Age distribution analysis
        ages = level['age']
        analysis['avg_age'] = ages.float().mean().item()
        analysis['max_age'] = ages.max().item()
        analysis['age_std'] = ages.float().std().item()
        
        # Salience distribution
        salience = level['salience']
        analysis['avg_salience'] = salience.mean().item()
        analysis['salience_std'] = salience.std().item()
        analysis['high_salience_ratio'] = (salience > 0.7).sum().item() / total_slots
        
        # Access pattern analysis
        access_counts = level['access_count']
        analysis['avg_access_count'] = access_counts.float().mean().item()
        analysis['access_std'] = access_counts.float().std().item()
        
        # Efficiency recommendations
        recommendations = []
        
        if utilization < 0.5:
            recommendations.append("Low utilization - consider reducing memory size")
        elif utilization > 0.95:
            recommendations.append("High utilization - consider increasing memory size")
        
        if analysis['avg_age'] > 1000:
            recommendations.append("High average age - increase eviction rate")
        
        if analysis['salience_std'] < 0.1:
            recommendations.append("Low salience variance - improve salience calculation")
        
        analysis['recommendations'] = recommendations
        
        return analysis


class ContextAwareEviction:
    """Context-aware eviction system that considers task and model state."""
    
    def __init__(self):
        self.task_memory_requirements = {
            'copy': {'working_memory': 0.3, 'long_term': 0.1},
            'lookup': {'working_memory': 0.4, 'long_term': 0.3},
            'reasoning': {'working_memory': 0.6, 'long_term': 0.5},
            'generation': {'working_memory': 0.5, 'long_term': 0.4}
        }
        
        self.current_task_context = None
    
    def set_task_context(self, task_type: str, difficulty: float = 0.5,
                        sequence_length: int = 0):
        """Set current task context for eviction decisions."""
        self.current_task_context = {
            'task_type': task_type,
            'difficulty': difficulty,
            'sequence_length': sequence_length,
            'timestamp': time.time()
        }
    
    def get_context_adjusted_policy(self, level_idx: int) -> Dict[str, float]:
        """Get context-adjusted eviction policy weights."""
        if not self.current_task_context:
            return {'lru_weight': 0.6, 'salience_weight': 0.4}
        
        task_type = self.current_task_context['task_type']
        difficulty = self.current_task_context['difficulty']
        
        # Get base requirements for task
        requirements = self.task_memory_requirements.get(task_type, 
                      {'working_memory': 0.5, 'long_term': 0.3})
        
        # Adjust based on memory level
        if level_idx == 0:  # Working memory level
            memory_importance = requirements['working_memory']
        else:  # Long-term memory levels
            memory_importance = requirements['long_term']
        
        # Adjust weights based on task requirements and difficulty
        salience_weight = memory_importance * (1.0 + difficulty * 0.5)
        lru_weight = 1.0 - salience_weight
        
        # Ensure weights are in valid range
        salience_weight = max(0.1, min(0.9, salience_weight))
        lru_weight = 1.0 - salience_weight
        
        return {
            'lru_weight': lru_weight,
            'salience_weight': salience_weight,
            'context_bonus': difficulty * 0.1
        }
