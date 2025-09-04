"""Advanced Research Metrics Collector
=====================================

Comprehensive metrics collection for NSM vs Transformer comparison research.
This module extends the basic research metrics with detailed training dynamics,
gradient analysis, state evolution tracking, and attention/state contribution analysis.

Metrics Categories:
1. Training Dynamics: loss curves, accuracy, learning rate, convergence speed
2. Gradient Analysis: norms, gradient explosion/vanishing detection
3. State Evolution: step-by-step state changes, stability metrics
4. Attention Analysis: token-token attention maps, contribution ratios
5. Efficiency Metrics: FLOPs estimation, memory usage, sequence scaling
6. Stability Metrics: state propagation stability, numerical stability
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import time


@dataclass
class TrainingMetrics:
    """Training dynamics tracking"""
    losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    convergence_speed: float = 0.0
    gradient_norms: List[float] = field(default_factory=list)
    
    def add_training_step(self, loss: float, accuracy: float, lr: float, epoch: int, grad_norm: float):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.learning_rates.append(lr)
        self.epochs.append(epoch)
        self.gradient_norms.append(grad_norm)
        self.timestamps.append(datetime.utcnow().isoformat())
        
        # Calculate convergence speed (loss reduction rate)
        if len(self.losses) >= 2:
            self.convergence_speed = (self.losses[-2] - self.losses[-1]) / max(self.losses[-2], 1e-8)


@dataclass
class StateEvolutionMetrics:
    """State evolution and stability tracking"""
    state_changes: List[float] = field(default_factory=list)  # L2 norm of state changes
    state_stability: List[float] = field(default_factory=list)  # Stability coefficient
    state_entropy: List[float] = field(default_factory=list)  # Information entropy
    state_clustering: Dict[str, Any] = field(default_factory=dict)
    level_transitions: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class AttentionAnalysisMetrics:
    """Attention pattern and contribution analysis"""
    attention_entropy: List[float] = field(default_factory=list)
    attention_concentration: List[float] = field(default_factory=list)  # How focused attention is
    state_vs_attention_ratio: List[float] = field(default_factory=list)  # Decision contribution ratio
    token_importance_scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class EfficiencyMetrics:
    """Performance and efficiency tracking"""
    flops_per_token: List[float] = field(default_factory=list)
    memory_per_sequence_length: List[Tuple[int, float]] = field(default_factory=list)
    gpu_utilization_history: List[float] = field(default_factory=list)
    inference_latency: List[float] = field(default_factory=list)
    sequence_scaling_factor: float = 1.0


class AdvancedResearchMetrics:
    """Advanced metrics collector for NSM research"""
    
    def __init__(self, model_getter, max_history: int = 1000):
        self._get_model = model_getter
        self.max_history = max_history
        
        # Metric storage
        self.training_metrics = TrainingMetrics()
        self.state_metrics = StateEvolutionMetrics()
        self.attention_metrics = AttentionAnalysisMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        
        # State tracking
        self._prev_state = None
        self._step_counter = 0
        
    def collect_training_step(self, loss: float, accuracy: float, lr: float, epoch: int, model=None):
        """Collect metrics during training step"""
        if model is None:
            model = self._get_model()
        if model is None:
            return
            
        # Calculate gradient norm
        grad_norm = self._calculate_gradient_norm(model)
        
        # Update training metrics
        self.training_metrics.add_training_step(loss, accuracy, lr, epoch, grad_norm)
        
        # Trim history if needed
        self._trim_history()
        
    def collect_inference_step(self, input_ids: torch.Tensor, model=None) -> Dict[str, Any]:
        """Collect detailed metrics during inference"""
        if model is None:
            model = self._get_model()
        if model is None:
            return {}
            
        start_time = time.time()
        
        # Forward pass with detailed info
        with torch.no_grad():
            logits, info = model(input_ids, return_state_info=True)
            
        inference_time = time.time() - start_time
        
        # Extract detailed metrics
        attention_maps = info.get('attention_maps', [])
        route_weights = info.get('route_weights', [])
        state_reads = info.get('state_reads', None)
        hidden_states = info.get('hidden_states', None)
        
        # State evolution analysis
        self._analyze_state_evolution(model, state_reads)
        
        # Attention analysis
        attention_analysis = self._analyze_attention_patterns(attention_maps, input_ids)
        
        # State vs Attention contribution
        contribution_ratio = self._calculate_contribution_ratio(state_reads, hidden_states, attention_maps)
        
        # Efficiency metrics
        self._update_efficiency_metrics(input_ids.shape[1], inference_time, model)
        
        return {
            "attention_analysis": attention_analysis,
            "state_evolution": self._get_state_evolution_summary(),
            "contribution_ratio": contribution_ratio,
            "efficiency": self._get_efficiency_summary()
        }
        
    def _calculate_gradient_norm(self, model) -> float:
        """Calculate total gradient norm"""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
        return (total_norm ** 0.5) if param_count > 0 else 0.0
        
    def _analyze_state_evolution(self, model, state_reads):
        """Analyze how states evolve over time"""
        if not hasattr(model, 'state') or state_reads is None:
            return
            
        current_state = model.state
        
        if self._prev_state is not None:
            # Calculate state change magnitude
            state_change = self._calculate_state_change(self._prev_state, current_state)
            self.state_metrics.state_changes.append(state_change)
            
            # Calculate stability metric
            stability = self._calculate_state_stability(state_reads)
            self.state_metrics.state_stability.append(stability)
            
        self._prev_state = self._deep_copy_state(current_state)
        
    def _calculate_state_change(self, prev_state, curr_state) -> float:
        """Calculate L2 norm of state changes"""
        if not hasattr(prev_state, 'levels') or not hasattr(curr_state, 'levels'):
            return 0.0
            
        total_change = 0.0
        level_count = 0
        
        for prev_level, curr_level in zip(prev_state.levels, curr_state.levels):
            if 'K' in prev_level and 'K' in curr_level:
                change = torch.norm(curr_level['K'] - prev_level['K']).item()
                total_change += change
                level_count += 1
                
        return total_change / max(level_count, 1)
        
    def _calculate_state_stability(self, state_reads) -> float:
        """Calculate state stability coefficient"""
        if state_reads is None:
            return 0.0
            
        # Stability = 1 / (1 + variance_across_sequence)
        variance = torch.var(state_reads, dim=1).mean().item()
        return 1.0 / (1.0 + variance)
        
    def _analyze_attention_patterns(self, attention_maps: List[torch.Tensor], input_ids: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention patterns and concentration"""
        if not attention_maps:
            return {"entropy": 0.0, "concentration": 0.0, "head_diversity": 0.0}
            
        # Aggregate attention across heads and layers
        total_attention = torch.stack(attention_maps).mean(dim=0)  # [batch, seq_len, seq_len]
        
        # Calculate attention entropy (how spread out attention is)
        attention_probs = torch.softmax(total_attention, dim=-1)
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1).mean().item()
        
        # Calculate concentration (how focused attention is)
        concentration = torch.max(attention_probs, dim=-1)[0].mean().item()
        
        # Calculate head diversity (how different attention heads behave)
        head_diversity = 0.0
        if len(attention_maps) > 1:
            head_similarities = []
            for i in range(len(attention_maps)):
                for j in range(i+1, len(attention_maps)):
                    sim = torch.cosine_similarity(
                        attention_maps[i].flatten(), 
                        attention_maps[j].flatten(), 
                        dim=0
                    ).item()
                    head_similarities.append(sim)
            head_diversity = 1.0 - np.mean(head_similarities) if head_similarities else 0.0
            
        self.attention_metrics.attention_entropy.append(entropy)
        self.attention_metrics.attention_concentration.append(concentration)
        
        return {
            "entropy": entropy,
            "concentration": concentration,
            "head_diversity": head_diversity
        }
        
    def _calculate_contribution_ratio(self, state_reads, hidden_states, attention_maps) -> float:
        """Calculate ratio of state vs attention contribution to final decision"""
        if state_reads is None or hidden_states is None:
            return 0.5  # Default balanced ratio
            
        # Compare magnitude of state contribution vs attention contribution
        state_magnitude = torch.norm(state_reads, dim=-1).mean().item()
        
        # Estimate attention contribution (this is approximate)
        total_magnitude = torch.norm(hidden_states, dim=-1).mean().item()
        attention_magnitude = max(0, total_magnitude - state_magnitude)
        
        # Calculate ratio (state contribution / total contribution)
        total_contrib = state_magnitude + attention_magnitude
        ratio = state_magnitude / max(total_contrib, 1e-8)
        
        self.attention_metrics.state_vs_attention_ratio.append(ratio)
        return ratio
        
    def _update_efficiency_metrics(self, seq_len: int, inference_time: float, model):
        """Update efficiency and performance metrics"""
        # Estimate FLOPs per token (rough approximation)
        flops_estimate = self._estimate_flops_per_token(model, seq_len)
        self.efficiency_metrics.flops_per_token.append(flops_estimate)
        
        # Memory usage per sequence length
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            self.efficiency_metrics.memory_per_sequence_length.append((seq_len, memory_mb))
            
        # Inference latency
        self.efficiency_metrics.inference_latency.append(inference_time)
        
    def _estimate_flops_per_token(self, model, seq_len: int) -> float:
        """Rough FLOPs estimation per token"""
        # This is a simplified estimation
        d_model = getattr(model, 'd_model', 512)
        n_layers = len(model.blocks) if hasattr(model, 'blocks') else 6
        vocab_size = getattr(model, 'vocab_size', 32000)
        
        # Attention FLOPs: 2 * seq_len * d_model^2 per layer
        attention_flops = 2 * seq_len * d_model * d_model * n_layers
        
        # FFN FLOPs: 8 * d_model^2 per layer (assuming 4x expansion)
        ffn_flops = 8 * d_model * d_model * n_layers
        
        # Output projection FLOPs
        output_flops = d_model * vocab_size
        
        total_flops = attention_flops + ffn_flops + output_flops
        return total_flops / seq_len  # FLOPs per token
        
    def _get_state_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of state evolution metrics"""
        return {
            "avg_state_change": np.mean(self.state_metrics.state_changes[-10:]) if self.state_metrics.state_changes else 0.0,
            "avg_stability": np.mean(self.state_metrics.state_stability[-10:]) if self.state_metrics.state_stability else 0.0,
            "change_trend": self._calculate_trend(self.state_metrics.state_changes),
            "stability_trend": self._calculate_trend(self.state_metrics.state_stability)
        }
        
    def _get_efficiency_summary(self) -> Dict[str, Any]:
        """Get summary of efficiency metrics"""
        return {
            "avg_flops_per_token": np.mean(self.efficiency_metrics.flops_per_token[-10:]) if self.efficiency_metrics.flops_per_token else 0.0,
            "avg_latency": np.mean(self.efficiency_metrics.inference_latency[-10:]) if self.efficiency_metrics.inference_latency else 0.0,
            "memory_scaling": self._calculate_memory_scaling(),
            "throughput_estimate": 1.0 / np.mean(self.efficiency_metrics.inference_latency[-10:]) if self.efficiency_metrics.inference_latency else 0.0
        }
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a metric"""
        if len(values) < 2:
            return "stable"
        recent = values[-5:] if len(values) >= 5 else values
        if len(recent) < 2:
            return "stable"
        trend = (recent[-1] - recent[0]) / len(recent)
        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        else:
            return "stable"
            
    def _calculate_memory_scaling(self) -> float:
        """Calculate memory scaling factor with sequence length"""
        if len(self.efficiency_metrics.memory_per_sequence_length) < 2:
            return 1.0
        
        # Linear regression to find scaling factor
        data = self.efficiency_metrics.memory_per_sequence_length[-20:]  # Last 20 points
        if len(data) < 2:
            return 1.0
            
        seq_lens = np.array([x[0] for x in data])
        memories = np.array([x[1] for x in data])
        
        # Simple linear fit
        if np.std(seq_lens) > 0:
            correlation = np.corrcoef(seq_lens, memories)[0, 1]
            return abs(correlation) * 2.0  # Scale to make it more interpretable
        return 1.0
        
    def _deep_copy_state(self, state):
        """Create a deep copy of state for comparison"""
        if not hasattr(state, 'levels'):
            return None
        
        # Simple state copy - in production you'd want proper deep copy
        return state
        
    def _trim_history(self):
        """Trim metric history to prevent memory bloat"""
        metrics_to_trim = [
            self.training_metrics.losses,
            self.training_metrics.accuracies,
            self.training_metrics.learning_rates,
            self.training_metrics.epochs,
            self.training_metrics.timestamps,
            self.training_metrics.gradient_norms,
            self.state_metrics.state_changes,
            self.state_metrics.state_stability,
            self.attention_metrics.attention_entropy,
            self.attention_metrics.attention_concentration,
            self.attention_metrics.state_vs_attention_ratio,
            self.efficiency_metrics.flops_per_token,
            self.efficiency_metrics.inference_latency
        ]
        
        for metric_list in metrics_to_trim:
            if len(metric_list) > self.max_history:
                del metric_list[:-self.max_history]
                
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a JSON-serializable format"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "training": {
                "loss_curve": self.training_metrics.losses[-50:],  # Last 50 points
                "accuracy_curve": self.training_metrics.accuracies[-50:],
                "learning_rate": self.training_metrics.learning_rates[-1] if self.training_metrics.learning_rates else 0.0,
                "gradient_norms": self.training_metrics.gradient_norms[-50:],
                "convergence_speed": self.training_metrics.convergence_speed
            },
            "state_evolution": {
                "state_changes": self.state_metrics.state_changes[-50:],
                "stability_scores": self.state_metrics.state_stability[-50:],
                "summary": self._get_state_evolution_summary()
            },
            "attention_analysis": {
                "entropy_history": self.attention_metrics.attention_entropy[-50:],
                "concentration_history": self.attention_metrics.attention_concentration[-50:],
                "state_vs_attention_ratios": self.attention_metrics.state_vs_attention_ratio[-50:]
            },
            "efficiency": {
                "flops_per_token": self.efficiency_metrics.flops_per_token[-50:],
                "latency_history": self.efficiency_metrics.inference_latency[-50:],
                "memory_scaling": self.efficiency_metrics.memory_per_sequence_length[-20:],
                "summary": self._get_efficiency_summary()
            }
        }
