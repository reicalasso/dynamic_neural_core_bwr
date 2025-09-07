"""
BWR-DNC 002: Research Tools

This module provides comprehensive research and analysis tools for the BWR-DNC model,
including visualization, metrics collection, and experimental features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from core.integration import MemoryIntegratedDNC
from memory.state_bank import StateBank
from utils import MetricsTracker, Logger


class MemoryAnalyzer:
    """
    Memory Analysis and Visualization Tools.
    
    Provides comprehensive analysis of the memory system including
    usage patterns, similarity analysis, and temporal dynamics.
    """
    
    def __init__(self, model: MemoryIntegratedDNC):
        """
        Initialize memory analyzer.
        
        Args:
            model: BWR-DNC model to analyze
        """
        self.model = model
        self.memory = model.memory
        self.logger = Logger("MemoryAnalyzer")
        
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """
        Analyze current memory usage patterns.
        
        Returns:
            Dictionary with usage analysis
        """
        stats = self.memory.get_memory_stats()
        
        analysis = {
            'utilization_by_level': {},
            'salience_distribution': {},
            'age_distribution': {},
            'access_patterns': {}
        }
        
        for level_idx, level in enumerate(self.memory.levels):
            level_name = f'level_{level_idx}'
            
            salience = level['salience'].detach().cpu().numpy()
            age = level['age'].detach().cpu().numpy()
            access_count = level['access_count'].detach().cpu().numpy()
            
            # Utilization analysis
            active_mask = salience > 0.1
            analysis['utilization_by_level'][level_name] = {
                'total_slots': len(salience),
                'active_slots': int(active_mask.sum()),
                'utilization_rate': float(active_mask.mean()),
                'avg_salience': float(salience.mean()),
                'max_salience': float(salience.max()),
                'min_salience': float(salience.min())
            }
            
            # Salience distribution
            analysis['salience_distribution'][level_name] = {
                'mean': float(salience.mean()),
                'std': float(salience.std()),
                'quantiles': {
                    'q25': float(np.percentile(salience, 25)),
                    'q50': float(np.percentile(salience, 50)),
                    'q75': float(np.percentile(salience, 75)),
                    'q95': float(np.percentile(salience, 95))
                }
            }
            
            # Age distribution
            analysis['age_distribution'][level_name] = {
                'mean_age': float(age.mean()),
                'max_age': int(age.max()),
                'median_age': float(np.median(age))
            }
            
            # Access patterns
            analysis['access_patterns'][level_name] = {
                'total_accesses': int(access_count.sum()),
                'avg_accesses': float(access_count.mean()),
                'most_accessed': int(access_count.max()),
                'unaccessed_slots': int((access_count == 0).sum())
            }
        
        return analysis
    
    def compute_memory_similarity_matrix(self, level_idx: int = 0) -> np.ndarray:
        """
        Compute similarity matrix for memory keys at a specific level.
        
        Args:
            level_idx: Memory level index
            
        Returns:
            Similarity matrix as numpy array
        """
        if level_idx >= len(self.memory.levels):
            raise ValueError(f"Level {level_idx} does not exist")
        
        keys = self.memory.levels[level_idx]['keys'].detach().cpu()
        
        # Compute cosine similarity
        keys_norm = keys / keys.norm(dim=1, keepdim=True)
        similarity_matrix = torch.mm(keys_norm, keys_norm.t())
        
        return similarity_matrix.numpy()
    
    def visualize_memory_heatmap(self, level_idx: int = 0, save_path: Optional[Path] = None):
        """
        Create heatmap visualization of memory similarity.
        
        Args:
            level_idx: Memory level index
            save_path: Optional path to save the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping visualization")
            return None
            
        similarity_matrix = self.compute_memory_similarity_matrix(level_idx)
        salience = self.memory.levels[level_idx]['salience'].detach().cpu().numpy()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Similarity heatmap
        im1 = ax1.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        ax1.set_title(f'Memory Similarity Matrix (Level {level_idx})')
        ax1.set_xlabel('Memory Slot')
        ax1.set_ylabel('Memory Slot')
        plt.colorbar(im1, ax=ax1)
        
        # Salience visualization
        sorted_indices = np.argsort(salience)[::-1]
        sorted_salience = salience[sorted_indices]
        
        ax2.bar(range(len(sorted_salience)), sorted_salience)
        ax2.set_title(f'Memory Salience Distribution (Level {level_idx})')
        ax2.set_xlabel('Memory Slot (sorted by salience)')
        ax2.set_ylabel('Salience')
        ax2.axhline(y=0.1, color='r', linestyle='--', label='Active threshold')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Memory heatmap saved to {save_path}")
        
        return fig
    
    def track_memory_dynamics(self, input_sequence: torch.Tensor, steps: int = 10):
        """
        Track memory dynamics over multiple forward passes.
        
        Args:
            input_sequence: Input sequence to process
            steps: Number of processing steps to track
            
        Returns:
            Dictionary with dynamics data
        """
        dynamics = {
            'step': [],
            'utilization': [],
            'avg_salience': [],
            'write_strength': [],
            'memory_similarity': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for step in range(steps):
                # Process input
                logits, metadata = self.model(input_sequence, return_memory_stats=True)
                
                # Record metrics
                memory_stats = metadata['memory_stats']
                dynamics['step'].append(step)
                dynamics['utilization'].append(memory_stats['utilization'])
                dynamics['avg_salience'].append(
                    np.mean([stats['avg_salience'] for key, stats in memory_stats.items() 
                            if key.startswith('level_')])
                )
                dynamics['write_strength'].append(metadata.get('write_strength', 0.0))
                
                # Compute average similarity
                similarities = []
                for level_idx in range(len(self.memory.levels)):
                    sim_matrix = self.compute_memory_similarity_matrix(level_idx)
                    # Average off-diagonal similarity
                    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
                    avg_sim = sim_matrix[mask].mean()
                    similarities.append(avg_sim)
                dynamics['memory_similarity'].append(np.mean(similarities))
                
                # Slightly modify input for next step
                input_sequence = torch.roll(input_sequence, shifts=1, dims=1)
        
        return dynamics
    
    def export_memory_state(self, filepath: Path):
        """
        Export current memory state to file.
        
        Args:
            filepath: Path to export file
        """
        memory_data = {}
        
        for level_idx, level in enumerate(self.memory.levels):
            memory_data[f'level_{level_idx}'] = {
                'keys': level['keys'].detach().cpu().numpy().tolist(),
                'values': level['values'].detach().cpu().numpy().tolist(),
                'salience': level['salience'].detach().cpu().numpy().tolist(),
                'age': level['age'].detach().cpu().numpy().tolist(),
                'access_count': level['access_count'].detach().cpu().numpy().tolist()
            }
        
        with open(filepath, 'w') as f:
            json.dump(memory_data, f, indent=2)
        
        self.logger.info(f"Memory state exported to {filepath}")


class AttentionAnalyzer:
    """
    Attention Pattern Analysis Tools.
    
    Analyzes attention patterns in the transformer blocks to understand
    information flow and processing patterns.
    """
    
    def __init__(self, model: MemoryIntegratedDNC):
        """
        Initialize attention analyzer.
        
        Args:
            model: BWR-DNC model to analyze
        """
        self.model = model
        self.attention_patterns = []
        self.hooks = []
        self.logger = Logger("AttentionAnalyzer")
        
    def register_attention_hooks(self):
        """Register hooks to capture attention patterns."""
        def attention_hook(module, input, output):
            # Store attention weights if available
            if hasattr(module, 'attn_weights'):
                self.attention_patterns.append(module.attn_weights.detach().cpu())
        
        # Register hooks on attention modules
        for layer_idx, block in enumerate(self.model.dnc.blocks):
            hook = block.self_attn.register_forward_hook(attention_hook)
            self.hooks.append(hook)
    
    def remove_attention_hooks(self):
        """Remove attention hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def analyze_attention_patterns(self, input_sequence: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze attention patterns for a given input.
        
        Args:
            input_sequence: Input sequence to analyze
            
        Returns:
            Dictionary with attention analysis
        """
        self.attention_patterns.clear()
        self.register_attention_hooks()
        
        try:
            # Process input
            self.model.eval()
            with torch.no_grad():
                _ = self.model(input_sequence)
            
            # Analyze patterns
            analysis = {
                'num_layers': len(self.attention_patterns),
                'attention_entropy': [],
                'attention_concentration': [],
                'head_specialization': []
            }
            
            for layer_idx, attn_weights in enumerate(self.attention_patterns):
                # attn_weights shape: [batch, num_heads, seq_len, seq_len]
                B, H, T, _ = attn_weights.shape
                
                # Compute entropy (measure of attention spread)
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
                analysis['attention_entropy'].append(entropy.mean().item())
                
                # Compute concentration (how focused attention is)
                max_attention = attn_weights.max(dim=-1)[0]
                analysis['attention_concentration'].append(max_attention.mean().item())
                
                # Compute head specialization (how different heads attend differently)
                head_similarities = []
                for h1 in range(H):
                    for h2 in range(h1 + 1, H):
                        sim = torch.cosine_similarity(
                            attn_weights[0, h1].flatten(),
                            attn_weights[0, h2].flatten(),
                            dim=0
                        )
                        head_similarities.append(sim.item())
                
                analysis['head_specialization'].append(1.0 - np.mean(head_similarities))
            
            return analysis
            
        finally:
            self.remove_attention_hooks()


class ModelProfiler:
    """
    Model Performance Profiler.
    
    Profiles model performance including inference time, memory usage,
    and computational efficiency.
    """
    
    def __init__(self, model: MemoryIntegratedDNC):
        """
        Initialize model profiler.
        
        Args:
            model: BWR-DNC model to profile
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.logger = Logger("ModelProfiler")
    
    def profile_inference_speed(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        sequence_lengths: List[int] = [32, 64, 128, 256],
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Profile inference speed across different input sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with profiling results
        """
        import time
        
        results = {
            'batch_size': [],
            'sequence_length': [],
            'avg_time': [],
            'tokens_per_second': [],
            'memory_usage_mb': []
        }
        
        self.model.eval()
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # Create test input
                input_ids = torch.randint(1, 1000, (batch_size, seq_len)).to(self.device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.model(input_ids)
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Measure inference time
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    
                    with torch.no_grad():
                        _ = self.model(input_ids)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    times.append(time.time() - start_time)
                
                avg_time = np.mean(times)
                tokens_per_second = (batch_size * seq_len) / avg_time
                
                # Measure memory usage
                memory_usage = 0
                if torch.cuda.is_available():
                    memory_usage = torch.cuda.memory_allocated() / 1024 / 1024
                
                results['batch_size'].append(batch_size)
                results['sequence_length'].append(seq_len)
                results['avg_time'].append(avg_time)
                results['tokens_per_second'].append(tokens_per_second)
                results['memory_usage_mb'].append(memory_usage)
                
                self.logger.info(f"Batch {batch_size}, Seq {seq_len}: "
                               f"{avg_time:.4f}s, {tokens_per_second:.0f} tok/s")
        
        return results
    
    def profile_memory_operations(self, num_runs: int = 50) -> Dict[str, Any]:
        """
        Profile memory read/write operations.
        
        Args:
            num_runs: Number of operations to profile
            
        Returns:
            Dictionary with memory operation timings
        """
        import time
        
        memory = self.model.memory
        d_model = memory.d_model
        
        # Test data
        queries = torch.randn(4, 32, d_model).to(self.device)
        writes = torch.randn(4, 32, d_model).to(self.device)
        
        # Profile read operations
        read_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = memory.read(queries)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            read_times.append(time.time() - start_time)
        
        # Profile write operations
        write_times = []
        for _ in range(num_runs):
            start_time = time.time()
            memory.write(writes)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            write_times.append(time.time() - start_time)
        
        return {
            'read_operations': {
                'avg_time': np.mean(read_times),
                'std_time': np.std(read_times),
                'min_time': np.min(read_times),
                'max_time': np.max(read_times)
            },
            'write_operations': {
                'avg_time': np.mean(write_times),
                'std_time': np.std(write_times),
                'min_time': np.min(write_times),
                'max_time': np.max(write_times)
            }
        }


def create_research_suite(model: MemoryIntegratedDNC) -> Dict[str, Any]:
    """
    Create a comprehensive research analysis suite.
    
    Args:
        model: BWR-DNC model to analyze
        
    Returns:
        Dictionary containing all research tools
    """
    return {
        'memory_analyzer': MemoryAnalyzer(model),
        'attention_analyzer': AttentionAnalyzer(model),
        'model_profiler': ModelProfiler(model)
    }


def run_comprehensive_analysis(
    model: MemoryIntegratedDNC,
    test_input: torch.Tensor,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run comprehensive analysis and save results.
    
    Args:
        model: BWR-DNC model to analyze
        test_input: Test input for analysis
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all analysis results
    """
    output_dir.mkdir(exist_ok=True)
    
    # Create research suite
    suite = create_research_suite(model)
    
    results = {}
    
    # Memory analysis
    print("Running memory analysis...")
    memory_analyzer = suite['memory_analyzer']
    results['memory_usage'] = memory_analyzer.analyze_memory_usage()
    results['memory_dynamics'] = memory_analyzer.track_memory_dynamics(test_input)
    
    # Save memory visualizations
    if MATPLOTLIB_AVAILABLE:
        fig = memory_analyzer.visualize_memory_heatmap(save_path=output_dir / "memory_heatmap.png")
        if fig:
            plt.close(fig)
    
    # Export memory state
    memory_analyzer.export_memory_state(output_dir / "memory_state.json")
    
    # Attention analysis
    print("Running attention analysis...")
    attention_analyzer = suite['attention_analyzer']
    results['attention_patterns'] = attention_analyzer.analyze_attention_patterns(test_input)
    
    # Performance profiling
    print("Running performance profiling...")
    profiler = suite['model_profiler']
    results['inference_speed'] = profiler.profile_inference_speed()
    results['memory_operations'] = profiler.profile_memory_operations()
    
    # Save all results
    with open(output_dir / "analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Analysis complete! Results saved to {output_dir}")
    return results
