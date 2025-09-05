import time
import psutil
import threading
import json
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import torch
import logging

try:
    import gpustat
    import nvidia_ml_py3 as nvml
    NVIDIA_AVAILABLE = True
    nvml.nvmlInit()
except ImportError:
    NVIDIA_AVAILABLE = False

@dataclass
class PerformanceMetrics:
    """Data structure for performance metrics."""
    timestamp: float
    # Training metrics
    tokens_per_second: float
    batch_time: float
    forward_time: float
    backward_time: float
    loss: float
    learning_rate: float
    
    # Memory metrics
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    memory_bandwidth_util: float
    
    # System metrics
    cpu_usage: float
    ram_usage: float
    temperature: float
    
    # Model-specific metrics
    attention_diversity: float
    state_bank_utilization: float
    compression_ratio: float
    memory_access_patterns: Dict[str, float]

class RTXPerformanceMonitor:
    """Advanced performance monitoring optimized for RTX GPUs."""
    
    def __init__(self, device='cuda:0', history_size=1000, update_interval=0.1):
        self.device = device
        self.device_id = int(device.split(':')[1]) if ':' in device else 0
        self.history_size = history_size
        self.update_interval = update_interval
        
        # Metrics storage
        self.metrics_history = deque(maxlen=history_size)
        self.real_time_metrics = {}
        
        # Performance counters
        self.training_counters = {
            'total_tokens': 0,
            'total_batches': 0,
            'total_training_time': 0.0
        }
        
        # Threading
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.logger = logging.getLogger(__name__)
        
        # Initialize NVIDIA monitoring if available
        if NVIDIA_AVAILABLE:
            try:
                self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(self.device_id)
                self.nvidia_available = True
            except Exception as e:
                self.logger.warning(f"Could not initialize NVIDIA monitoring: {e}")
                self.nvidia_available = False
        else:
            self.nvidia_available = False
    
    def start_monitoring(self):
        """Start background performance monitoring."""
        if self.monitoring_thread is not None:
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring_thread is None:
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join()
        self.monitoring_thread = None
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self.stop_monitoring.wait(self.update_interval):
            try:
                metrics = self._collect_system_metrics()
                self.real_time_metrics.update(metrics)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        metrics = {}
        
        # CPU and memory
        metrics['cpu_usage'] = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        metrics['ram_usage'] = memory.percent
        metrics['ram_used_gb'] = memory.used / (1024**3)
        
        # GPU metrics (NVIDIA)
        if self.nvidia_available:
            try:
                # Memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                metrics['gpu_memory_used'] = mem_info.used / (1024**3)  # GB
                metrics['gpu_memory_total'] = mem_info.total / (1024**3)  # GB
                metrics['gpu_memory_percent'] = (mem_info.used / mem_info.total) * 100
                
                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                metrics['gpu_utilization'] = util.gpu
                metrics['memory_bandwidth_util'] = util.memory
                
                # Temperature
                temp = nvml.nvmlDeviceGetTemperature(self.gpu_handle, nvml.NVML_TEMPERATURE_GPU)
                metrics['temperature'] = temp
                
                # Power
                power = nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Watts
                metrics['power_usage'] = power
                
                # Clock speeds
                graphics_clock = nvml.nvmlDeviceGetClockInfo(self.gpu_handle, nvml.NVML_CLOCK_GRAPHICS)
                memory_clock = nvml.nvmlDeviceGetClockInfo(self.gpu_handle, nvml.NVML_CLOCK_MEM)
                metrics['graphics_clock'] = graphics_clock
                metrics['memory_clock'] = memory_clock
                
            except Exception as e:
                self.logger.warning(f"Error collecting GPU metrics: {e}")
        
        # PyTorch GPU metrics
        if torch.cuda.is_available():
            try:
                metrics['torch_memory_allocated'] = torch.cuda.memory_allocated(self.device) / (1024**3)
                metrics['torch_memory_reserved'] = torch.cuda.memory_reserved(self.device) / (1024**3)
                metrics['torch_memory_cached'] = torch.cuda.memory_cached(self.device) / (1024**3)
            except Exception as e:
                self.logger.warning(f"Error collecting PyTorch GPU metrics: {e}")
        
        return metrics
    
    def record_training_step(self, batch_size: int, forward_time: float, 
                           backward_time: float, loss: float, lr: float,
                           model_info: Optional[Dict] = None):
        """Record metrics for a training step."""
        timestamp = time.time()
        batch_time = forward_time + backward_time
        tokens_per_second = batch_size / batch_time if batch_time > 0 else 0
        
        # Collect current system metrics
        current_metrics = self.real_time_metrics.copy()
        
        # Create performance metrics object
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            tokens_per_second=tokens_per_second,
            batch_time=batch_time,
            forward_time=forward_time,
            backward_time=backward_time,
            loss=loss,
            learning_rate=lr,
            gpu_memory_used=current_metrics.get('gpu_memory_used', 0),
            gpu_memory_total=current_metrics.get('gpu_memory_total', 0),
            gpu_utilization=current_metrics.get('gpu_utilization', 0),
            memory_bandwidth_util=current_metrics.get('memory_bandwidth_util', 0),
            cpu_usage=current_metrics.get('cpu_usage', 0),
            ram_usage=current_metrics.get('ram_usage', 0),
            temperature=current_metrics.get('temperature', 0),
            attention_diversity=model_info.get('attention_diversity', 0) if model_info else 0,
            state_bank_utilization=model_info.get('state_bank_utilization', 0) if model_info else 0,
            compression_ratio=model_info.get('compression_ratio', 1.0) if model_info else 1.0,
            memory_access_patterns=model_info.get('memory_access_patterns', {}) if model_info else {}
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Update counters
        self.training_counters['total_tokens'] += batch_size
        self.training_counters['total_batches'] += 1
        self.training_counters['total_training_time'] += batch_time
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        return self.real_time_metrics.copy()
    
    def get_training_summary(self, last_n_steps: int = 100) -> Dict[str, Any]:
        """Get training performance summary."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-last_n_steps:]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        avg_metrics = {}
        for field in ['tokens_per_second', 'batch_time', 'forward_time', 'backward_time',
                     'gpu_memory_used', 'gpu_utilization', 'memory_bandwidth_util',
                     'cpu_usage', 'temperature']:
            values = [getattr(m, field) for m in recent_metrics if getattr(m, field, 0) > 0]
            if values:
                avg_metrics[f'avg_{field}'] = sum(values) / len(values)
                avg_metrics[f'max_{field}'] = max(values)
                avg_metrics[f'min_{field}'] = min(values)
        
        # Training efficiency metrics
        total_time = sum(m.batch_time for m in recent_metrics)
        total_tokens = sum(m.tokens_per_second * m.batch_time for m in recent_metrics)
        
        if total_time > 0:
            avg_metrics['overall_tokens_per_second'] = total_tokens / total_time
            avg_metrics['training_efficiency'] = avg_metrics.get('avg_gpu_utilization', 0) / 100.0
        
        # Memory efficiency
        if avg_metrics.get('max_gpu_memory_used', 0) > 0:
            avg_metrics['memory_efficiency'] = avg_metrics.get('avg_tokens_per_second', 0) / avg_metrics['max_gpu_memory_used']
        
        return avg_metrics
    
    def analyze_performance_bottlenecks(self) -> Dict[str, str]:
        """Analyze performance and identify bottlenecks."""
        if len(self.metrics_history) < 10:
            return {"status": "Insufficient data for analysis"}
        
        recent_metrics = list(self.metrics_history)[-50:]
        bottlenecks = {}
        
        # GPU utilization analysis
        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        if avg_gpu_util < 70:
            bottlenecks['gpu_underutilization'] = f"GPU utilization is low ({avg_gpu_util:.1f}%). Consider increasing batch size or model size."
        
        # Memory analysis
        avg_memory_util = sum(m.gpu_memory_used for m in recent_metrics if m.gpu_memory_used > 0)
        if avg_memory_util > 0:
            avg_memory_util /= len([m for m in recent_metrics if m.gpu_memory_used > 0])
            total_memory = recent_metrics[-1].gpu_memory_total
            
            if total_memory > 0:
                memory_percent = (avg_memory_util / total_memory) * 100
                if memory_percent > 90:
                    bottlenecks['memory_pressure'] = f"GPU memory usage is high ({memory_percent:.1f}%). Consider reducing batch size or enabling gradient checkpointing."
                elif memory_percent < 50:
                    bottlenecks['memory_underutilization'] = f"GPU memory usage is low ({memory_percent:.1f}%). Consider increasing batch size or model size."
        
        # Temperature analysis
        avg_temp = sum(m.temperature for m in recent_metrics if m.temperature > 0)
        if avg_temp > 0:
            avg_temp /= len([m for m in recent_metrics if m.temperature > 0])
            if avg_temp > 80:
                bottlenecks['thermal_throttling'] = f"GPU temperature is high ({avg_temp:.1f}°C). Check cooling and consider reducing clock speeds."
        
        # Batch time analysis
        forward_times = [m.forward_time for m in recent_metrics if m.forward_time > 0]
        backward_times = [m.backward_time for m in recent_metrics if m.backward_time > 0]
        
        if forward_times and backward_times:
            avg_forward = sum(forward_times) / len(forward_times)
            avg_backward = sum(backward_times) / len(backward_times)
            
            if avg_backward > avg_forward * 1.5:
                bottlenecks['backward_bottleneck'] = f"Backward pass is slower than forward ({avg_backward:.3f}s vs {avg_forward:.3f}s). Consider gradient checkpointing or mixed precision."
        
        return bottlenecks if bottlenecks else {"status": "No significant bottlenecks detected"}
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file."""
        if format == 'json':
            data = {
                'metrics_history': [asdict(m) for m in self.metrics_history],
                'training_counters': self.training_counters,
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'csv':
            import csv
            
            if not self.metrics_history:
                return
            
            with open(filepath, 'w', newline='') as f:
                fieldnames = list(asdict(self.metrics_history[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for metrics in self.metrics_history:
                    writer.writerow(asdict(metrics))
    
    def get_tensorboard_scalars(self, step: int) -> Dict[str, float]:
        """Get metrics formatted for TensorBoard logging."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        scalars = {
            'performance/tokens_per_second': latest.tokens_per_second,
            'performance/batch_time': latest.batch_time,
            'performance/forward_time': latest.forward_time,
            'performance/backward_time': latest.backward_time,
            'memory/gpu_memory_used_gb': latest.gpu_memory_used,
            'memory/gpu_utilization_percent': latest.gpu_utilization,
            'memory/memory_bandwidth_util': latest.memory_bandwidth_util,
            'system/cpu_usage': latest.cpu_usage,
            'system/temperature': latest.temperature,
            'training/loss': latest.loss,
            'training/learning_rate': latest.learning_rate,
            'model/attention_diversity': latest.attention_diversity,
            'model/state_bank_utilization': latest.state_bank_utilization,
            'model/compression_ratio': latest.compression_ratio
        }
        
        return scalars


class ModelPerformanceAnalyzer:
    """Analyze model-specific performance characteristics."""
    
    def __init__(self):
        self.attention_patterns = deque(maxlen=1000)
        self.memory_access_stats = defaultdict(list)
    
    def analyze_attention_diversity(self, attention_maps: List[torch.Tensor]) -> float:
        """Calculate attention diversity score."""
        if not attention_maps:
            return 0.0
        
        diversities = []
        for attn_map in attention_maps:
            if attn_map.numel() > 0:
                # Calculate entropy as diversity measure
                attn_flat = attn_map.flatten()
                attn_prob = torch.softmax(attn_flat, dim=0)
                entropy = -torch.sum(attn_prob * torch.log(attn_prob + 1e-8))
                diversities.append(entropy.item())
        
        return sum(diversities) / len(diversities) if diversities else 0.0
    
    def analyze_state_bank_utilization(self, state_bank) -> float:
        """Calculate state bank utilization efficiency."""
        total_utilization = 0.0
        total_levels = 0
        
        for level in state_bank.levels:
            salience = level['salience']
            active_slots = (salience > 0.1).sum().float()
            total_slots = salience.numel()
            
            if total_slots > 0:
                level_utilization = active_slots / total_slots
                total_utilization += level_utilization
                total_levels += 1
        
        return total_utilization / total_levels if total_levels > 0 else 0.0
    
    def calculate_compression_ratio(self, state_bank) -> float:
        """Calculate effective compression ratio across memory levels."""
        total_original_size = 0
        total_compressed_size = 0
        
        for level_idx, level in enumerate(state_bank.levels):
            original_size = level['V'].numel()
            
            # Estimate compressed size based on compression ratio
            compression_factor = 2 ** level_idx
            compressed_size = original_size / compression_factor
            
            total_original_size += original_size
            total_compressed_size += compressed_size
        
        return total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0
