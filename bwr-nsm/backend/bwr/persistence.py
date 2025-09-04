import torch
import os
import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
import logging

class StatePersistenceManager:
    """Advanced state persistence for cross-session memory."""
    
    def __init__(self, base_dir="./state_cache", max_cache_size_gb=1.0):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024**3  # Convert to bytes
        self.session_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        
        # Create subdirectories
        (self.base_dir / "sessions").mkdir(exist_ok=True)
        (self.base_dir / "global").mkdir(exist_ok=True)
        (self.base_dir / "checkpoints").mkdir(exist_ok=True)
        
    def save_state_bank(self, state_bank, session_id=None, global_save=False):
        """Save state bank with metadata."""
        session_id = session_id or self.session_id
        timestamp = datetime.now().isoformat()
        
        # Prepare state data
        state_data = {
            'timestamp': timestamp,
            'session_id': session_id,
            'levels': []
        }
        
        # Save each memory level
        for level_idx, level in enumerate(state_bank.levels):
            level_data = {
                'K': level['K'].detach().cpu(),
                'V': level['V'].detach().cpu(),
                'salience': level['salience'].detach().cpu(),
                'age': level['age'].detach().cpu(),
                'access_count': level['access_count'].detach().cpu()
            }
            state_data['levels'].append(level_data)
        
        # Save compressor states
        state_data['compressor_states'] = []
        for compressor in state_bank.compressors:
            compressor_state = compressor.state_dict()
            # Convert to CPU
            for key in compressor_state:
                compressor_state[key] = compressor_state[key].cpu()
            state_data['compressor_states'].append(compressor_state)
        
        # Choose save location
        if global_save:
            save_path = self.base_dir / "global" / f"state_{timestamp}.pt"
        else:
            session_dir = self.base_dir / "sessions" / session_id
            session_dir.mkdir(exist_ok=True)
            save_path = session_dir / f"state_{timestamp}.pt"
        
        # Save with compression
        torch.save(state_data, save_path, _use_new_zipfile_serialization=True)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'session_id': session_id,
            'file_path': str(save_path),
            'levels_count': len(state_data['levels']),
            'total_slots': sum(level['K'].shape[0] for level in state_data['levels']),
            'file_size': save_path.stat().st_size
        }
        
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"State saved to {save_path}")
        self._cleanup_old_states()
        
        return str(save_path)
    
    def load_state_bank(self, state_bank, session_id=None, latest_global=False):
        """Load state bank from saved data."""
        session_id = session_id or self.session_id
        
        if latest_global:
            # Load latest global state
            global_dir = self.base_dir / "global"
            state_files = list(global_dir.glob("state_*.pt"))
        else:
            # Load from specific session
            session_dir = self.base_dir / "sessions" / session_id
            if not session_dir.exists():
                self.logger.warning(f"Session {session_id} not found")
                return False
            state_files = list(session_dir.glob("state_*.pt"))
        
        if not state_files:
            self.logger.warning("No state files found")
            return False
        
        # Get latest file
        latest_file = max(state_files, key=lambda p: p.stat().st_mtime)
        
        try:
            state_data = torch.load(latest_file, map_location='cpu')
            
            # Restore memory levels
            device = next(state_bank.parameters()).device
            for level_idx, level_data in enumerate(state_data['levels']):
                if level_idx < len(state_bank.levels):
                    level = state_bank.levels[level_idx]
                    level['K'].data = level_data['K'].to(device)
                    level['V'].data = level_data['V'].to(device)
                    level['salience'].data = level_data['salience'].to(device)
                    level['age'].data = level_data['age'].to(device)
                    level['access_count'].data = level_data['access_count'].to(device)
            
            # Restore compressor states
            if 'compressor_states' in state_data:
                for comp_idx, comp_state in enumerate(state_data['compressor_states']):
                    if comp_idx < len(state_bank.compressors):
                        # Move to device
                        for key in comp_state:
                            comp_state[key] = comp_state[key].to(device)
                        state_bank.compressors[comp_idx].load_state_dict(comp_state)
            
            self.logger.info(f"State loaded from {latest_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False
    
    def save_model_checkpoint(self, model, optimizer, epoch, step, metrics):
        """Save complete model checkpoint."""
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        
        checkpoint_path = self.base_dir / "checkpoints" / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'timestamp': checkpoint_data['timestamp'],
            'file_path': str(checkpoint_path),
            'file_size': checkpoint_path.stat().st_size
        }
        
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(checkpoint_path)
    
    def get_session_history(self, session_id=None):
        """Get history of saves for a session."""
        session_id = session_id or self.session_id
        session_dir = self.base_dir / "sessions" / session_id
        
        if not session_dir.exists():
            return []
        
        history = []
        for metadata_file in session_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                history.append(metadata)
            except Exception as e:
                self.logger.warning(f"Error reading metadata {metadata_file}: {e}")
        
        return sorted(history, key=lambda x: x['timestamp'], reverse=True)
    
    def _cleanup_old_states(self):
        """Clean up old states to maintain cache size limit."""
        total_size = 0
        all_files = []
        
        # Collect all state files with their sizes
        for root in [self.base_dir / "sessions", self.base_dir / "global"]:
            if root.exists():
                for file_path in root.rglob("*.pt"):
                    size = file_path.stat().st_size
                    mtime = file_path.stat().st_mtime
                    all_files.append((file_path, size, mtime))
                    total_size += size
        
        # Remove oldest files if over limit
        if total_size > self.max_cache_size:
            # Sort by modification time (oldest first)
            all_files.sort(key=lambda x: x[2])
            
            for file_path, size, _ in all_files:
                if total_size <= self.max_cache_size:
                    break
                
                try:
                    file_path.unlink()
                    # Also remove metadata file
                    metadata_path = file_path.with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    total_size -= size
                    self.logger.info(f"Cleaned up old state: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Error cleaning up {file_path}: {e}")

    def create_user_profile(self, user_id, preferences=None):
        """Create persistent user profile."""
        user_dir = self.base_dir / "users" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        profile = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'preferences': preferences or {},
            'interaction_history': [],
            'model_adaptations': {}
        }
        
        profile_path = user_dir / "profile.json"
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        return profile_path
    
    def update_user_interaction(self, user_id, interaction_data):
        """Update user interaction history."""
        user_dir = self.base_dir / "users" / user_id
        profile_path = user_dir / "profile.json"
        
        if profile_path.exists():
            with open(profile_path) as f:
                profile = json.load(f)
            
            profile['interaction_history'].append({
                'timestamp': datetime.now().isoformat(),
                'data': interaction_data
            })
            
            # Keep only last 1000 interactions
            profile['interaction_history'] = profile['interaction_history'][-1000:]
            
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
