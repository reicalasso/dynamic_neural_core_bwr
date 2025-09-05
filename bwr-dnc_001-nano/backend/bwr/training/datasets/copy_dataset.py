"""
Dataset implementations for DNC training.
"""

import torch
from .base_dataset import BaseDataset

class SimpleCopyDataset(BaseDataset):
    """A simple dataset for the copy task."""
    
    def __init__(self, num_samples=1000, seq_len=16, vocab_size=50):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate copy task data."""
        data = []
        for _ in range(self.num_samples):
            # Generate random sequence with smaller vocabulary
            sequence = torch.randint(1, self.vocab_size, (self.seq_len,))
            # Add start and end tokens
            start_token = torch.tensor([self.vocab_size])  # Special start token
            end_token = torch.tensor([self.vocab_size + 1])  # Special end token
            
            # Input: start_token + sequence
            # Target: sequence + end_token
            input_seq = torch.cat([start_token, sequence])
            target_seq = torch.cat([sequence, end_token])
            
            data.append((input_seq, target_seq))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate function to handle variable length sequences."""
    inputs, targets = zip(*batch)
    
    # Pad sequences to the same length
    max_input_len = max([inp.shape[0] for inp in inputs])
    max_target_len = max([tgt.shape[0] for tgt in targets])
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        # Pad input
        input_padding = torch.zeros(max_input_len - inp.shape[0], dtype=torch.long)
        padded_input = torch.cat([inp, input_padding])
        padded_inputs.append(padded_input)
        
        # Pad target
        target_padding = torch.zeros(max_target_len - tgt.shape[0], dtype=torch.long)
        padded_target = torch.cat([tgt, target_padding])
        padded_targets.append(padded_target)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)