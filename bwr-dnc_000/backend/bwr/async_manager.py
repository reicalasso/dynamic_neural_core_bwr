import asyncio
import threading
import queue
import time
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

@dataclass
class MemoryUpdate:
    """Data structure for memory update operations."""
    level: int
    data: torch.Tensor
    priority: float
    timestamp: float
    metadata: Dict[str, Any]

class AsyncMemoryManager:
    """Asynchronous memory management for background state updates."""
    
    def __init__(self, state_bank, max_workers=4, update_interval=0.1):
        self.state_bank = state_bank
        self.max_workers = max_workers
        self.update_interval = update_interval
        
        # Async components
        self.update_queue = asyncio.Queue(maxsize=1000)
        self.consolidation_queue = asyncio.Queue(maxsize=100)
        self.is_running = False
        self.background_tasks = []
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance metrics
        self.metrics = {
            'updates_processed': 0,
            'consolidations_done': 0,
            'queue_overflow_count': 0,
            'average_update_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start background processing tasks."""
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._update_processor()),
            asyncio.create_task(self._consolidation_processor()),
            asyncio.create_task(self._periodic_cleanup())
        ]
        
        self.logger.info("Async memory manager started")
    
    async def stop(self):
        """Stop background processing."""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Async memory manager stopped")
    
    async def queue_memory_update(self, data: torch.Tensor, level: int = 0, 
                                 priority: float = 1.0, metadata: Dict = None):
        """Queue a memory update operation."""
        update = MemoryUpdate(
            level=level,
            data=data.detach().clone(),
            priority=priority,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        try:
            await self.update_queue.put(update)
        except asyncio.QueueFull:
            self.metrics['queue_overflow_count'] += 1
            self.logger.warning("Memory update queue full, dropping update")
    
    async def _update_processor(self):
        """Background task to process memory updates."""
        while self.is_running:
            try:
                # Wait for updates with timeout
                update = await asyncio.wait_for(
                    self.update_queue.get(), 
                    timeout=self.update_interval
                )
                
                start_time = time.time()
                await self._process_update(update)
                
                # Update metrics
                update_time = time.time() - start_time
                self.metrics['updates_processed'] += 1
                self.metrics['average_update_time'] = (
                    (self.metrics['average_update_time'] * (self.metrics['updates_processed'] - 1) + 
                     update_time) / self.metrics['updates_processed']
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing memory update: {e}")
    
    async def _process_update(self, update: MemoryUpdate):
        """Process a single memory update."""
        loop = asyncio.get_event_loop()
        
        # Run memory update in thread pool to avoid blocking
        await loop.run_in_executor(
            self.executor,
            self._update_memory_level,
            update
        )
    
    def _update_memory_level(self, update: MemoryUpdate):
        """Update memory level (runs in thread pool)."""
        try:
            level = self.state_bank.levels[update.level]
            
            # Compute salience score
            salience_score = self._compute_salience(update.data, update.metadata)
            
            # Find eviction candidate
            evict_idx = self._find_eviction_candidate(level, update.priority)
            
            # Update memory
            with torch.no_grad():
                level['K'][evict_idx] = update.data.to(level['K'].device)
                level['V'][evict_idx] = update.data.to(level['V'].device)
                level['salience'][evict_idx] = salience_score
                level['age'][evict_idx] = 0
                level['access_count'][evict_idx] += 1
                
                # Age other slots
                mask = torch.ones_like(level['age'], dtype=torch.bool)
                mask[evict_idx] = False
                level['age'][mask] += 1
        
        except Exception as e:
            self.logger.error(f"Error updating memory level {update.level}: {e}")
    
    def _compute_salience(self, data: torch.Tensor, metadata: Dict) -> float:
        """Compute salience score for memory update."""
        base_salience = metadata.get('importance', 0.5)
        
        # Add recency bonus
        recency_bonus = min(0.3, 1.0 / (time.time() - metadata.get('timestamp', time.time()) + 1))
        
        # Add attention bonus if available
        attention_bonus = metadata.get('attention_weight', 0.0) * 0.2
        
        return min(1.0, base_salience + recency_bonus + attention_bonus)
    
    def _find_eviction_candidate(self, level: Dict, priority: float) -> int:
        """Find best candidate for eviction using LRU + Salience."""
        # Combine age and inverse salience for eviction score
        age_score = level['age'].float() / (level['age'].max() + 1e-8)
        salience_score = 1.0 - level['salience']
        
        # Weighted combination (higher score = better eviction candidate)
        eviction_score = 0.6 * age_score + 0.4 * salience_score
        
        # Add randomness to avoid always evicting the same slot
        noise = torch.rand_like(eviction_score) * 0.1
        final_score = eviction_score + noise
        
        return torch.argmax(final_score).item()
    
    async def _consolidation_processor(self):
        """Background task for memory consolidation."""
        while self.is_running:
            try:
                # Periodic consolidation
                await asyncio.sleep(5.0)  # Consolidate every 5 seconds
                await self._perform_consolidation()
                
            except Exception as e:
                self.logger.error(f"Error in consolidation: {e}")
    
    async def _perform_consolidation(self):
        """Perform memory consolidation across levels."""
        loop = asyncio.get_event_loop()
        
        await loop.run_in_executor(
            self.executor,
            self._consolidate_memory_levels
        )
        
        self.metrics['consolidations_done'] += 1
    
    def _consolidate_memory_levels(self):
        """Consolidate memory from lower to higher levels."""
        try:
            for level_idx in range(len(self.state_bank.levels) - 1):
                source_level = self.state_bank.levels[level_idx]
                target_level = self.state_bank.levels[level_idx + 1]
                
                # Find high-salience, old memories to promote
                age_threshold = source_level['age'].quantile(0.8)
                salience_threshold = source_level['salience'].quantile(0.7)
                
                candidates = (
                    (source_level['age'] > age_threshold) & 
                    (source_level['salience'] > salience_threshold)
                )
                
                if candidates.any():
                    # Select best candidate
                    candidate_idx = torch.argmax(
                        source_level['salience'][candidates]
                    ).item()
                    
                    # Find indices in original tensor
                    candidate_indices = torch.where(candidates)[0]
                    actual_idx = candidate_indices[candidate_idx].item()
                    
                    # Compress and move to next level
                    compressed_data = self.state_bank.compressors[level_idx + 1](
                        source_level['V'][actual_idx:actual_idx+1].unsqueeze(0),
                        compress=True
                    )
                    
                    # Find eviction target in higher level
                    target_evict_idx = torch.argmin(target_level['salience']).item()
                    
                    # Move data
                    with torch.no_grad():
                        decompressed = self.state_bank.compressors[level_idx + 1](
                            compressed_data, compress=False
                        ).squeeze(0)
                        
                        target_level['K'][target_evict_idx] = decompressed
                        target_level['V'][target_evict_idx] = decompressed
                        target_level['salience'][target_evict_idx] = source_level['salience'][actual_idx] * 0.9
                        target_level['age'][target_evict_idx] = 0
                        
                        # Clear source slot
                        source_level['salience'][actual_idx] = 0.0
                        source_level['age'][actual_idx] = 0
        
        except Exception as e:
            self.logger.error(f"Error in memory consolidation: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of stale memory."""
        while self.is_running:
            try:
                await asyncio.sleep(30.0)  # Cleanup every 30 seconds
                await self._cleanup_stale_memory()
                
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
    
    async def _cleanup_stale_memory(self):
        """Clean up very old, low-salience memories."""
        loop = asyncio.get_event_loop()
        
        await loop.run_in_executor(
            self.executor,
            self._perform_cleanup
        )
    
    def _perform_cleanup(self):
        """Perform memory cleanup (runs in thread pool)."""
        try:
            for level in self.state_bank.levels:
                # Find very old, low-salience memories
                old_threshold = level['age'].quantile(0.95)
                low_salience_threshold = level['salience'].quantile(0.1)
                
                cleanup_candidates = (
                    (level['age'] > old_threshold) & 
                    (level['salience'] < low_salience_threshold)
                )
                
                if cleanup_candidates.any():
                    with torch.no_grad():
                        level['salience'][cleanup_candidates] = 0.0
                        level['age'][cleanup_candidates] = 0
                        level['access_count'][cleanup_candidates] = 0
        
        except Exception as e:
            self.logger.error(f"Error in memory cleanup: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            **self.metrics,
            'queue_size': self.update_queue.qsize(),
            'consolidation_queue_size': self.consolidation_queue.qsize(),
            'is_running': self.is_running
        }


class AsyncTrainingManager:
    """Asynchronous training management for non-blocking operations."""
    
    def __init__(self, model, memory_manager: AsyncMemoryManager):
        self.model = model
        self.memory_manager = memory_manager
        self.training_tasks = []
        self.is_training = False
        self.logger = logging.getLogger(__name__)
    
    async def start_async_training(self, dataloader, optimizer, config):
        """Start asynchronous training with background memory updates."""
        self.is_training = True
        
        # Start memory manager
        await self.memory_manager.start()
        
        try:
            await self._training_loop(dataloader, optimizer, config)
        finally:
            await self.memory_manager.stop()
            self.is_training = False
    
    async def _training_loop(self, dataloader, optimizer, config):
        """Main training loop with async memory updates."""
        for epoch in range(config.get('epochs', 10)):
            for batch_idx, batch in enumerate(dataloader):
                if not self.is_training:
                    break
                
                # Forward pass
                inputs, targets = batch[0], batch[1]
                logits, aux_info = self.model(inputs.cuda(), return_state_info=True)
                
                # Compute loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.cuda().view(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Queue async memory updates
                if aux_info and 'hidden_states' in aux_info:
                    await self.memory_manager.queue_memory_update(
                        data=aux_info['hidden_states'].mean(dim=1),  # Pool over sequence
                        level=0,
                        priority=1.0 - loss.item(),  # Higher priority for lower loss
                        metadata={
                            'epoch': epoch,
                            'batch': batch_idx,
                            'loss': loss.item(),
                            'timestamp': time.time()
                        }
                    )
                
                # Yield control periodically
                if batch_idx % 10 == 0:
                    await asyncio.sleep(0.001)  # Allow other tasks to run
    
    async def stop_training(self):
        """Stop training gracefully."""
        self.is_training = False
        await self.memory_manager.stop()
