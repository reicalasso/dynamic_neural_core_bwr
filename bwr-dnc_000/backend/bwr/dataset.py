import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, Dataset
import json

class LongRangeDataset(Dataset):
    """Advanced dataset for long-range reasoning tasks."""
    
    def __init__(self, task='copy', num_samples=1000, seq_len=128, vocab_size=1000):
        self.task = task
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self._generate_data()
    
    def _generate_data(self):
        if self.task == 'copy':
            return self._generate_copy_task()
        elif self.task == 'lookup':
            return self._generate_lookup_task()
        elif self.task == 'long_infill':
            return self._generate_long_infill_task()
        elif self.task == 'needle_haystack':
            return self._generate_needle_haystack()
        elif self.task == 'associative_recall':
            return self._generate_associative_recall()
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _generate_copy_task(self):
        """Simple copy task with increasing difficulty."""
        data = []
        for i in range(self.num_samples):
            # Progressive difficulty: start small, increase sequence length
            difficulty = min(1.0, i / (self.num_samples * 0.8))
            current_len = int(self.seq_len * (0.3 + 0.7 * difficulty))
            
            sequence = torch.randint(1, self.vocab_size, (current_len,))
            # Pad to fixed length
            padded_seq = torch.cat([sequence, torch.zeros(self.seq_len - current_len, dtype=torch.long)])
            
            inputs = padded_seq[:-1]
            targets = padded_seq[1:]
            
            data.append((inputs, targets))
        return data
    
    def _generate_lookup_task(self):
        """Key-value lookup over long sequences."""
        data = []
        for _ in range(self.num_samples):
            sequence = []
            kv_pairs = {}
            
            # Create key-value pairs
            num_pairs = random.randint(5, 15)
            for _ in range(num_pairs):
                key = random.randint(10, 99)  # 2-digit keys
                value = random.randint(100, 999)  # 3-digit values
                kv_pairs[key] = value
                
                # Add to sequence: KEY <sep> VALUE <sep>
                sequence.extend([key, 1, value, 1])  # 1 is separator
            
            # Add distractor tokens
            distractor_len = self.seq_len - len(sequence) - 10
            if distractor_len > 0:
                distractors = torch.randint(1000, self.vocab_size, (distractor_len,)).tolist()
                sequence.extend(distractors)
            
            # Add query: KEY <sep> ?
            query_key = random.choice(list(kv_pairs.keys()))
            sequence.extend([query_key, 1, 0])  # 0 is query token
            
            # Pad sequence
            if len(sequence) < self.seq_len:
                sequence.extend([0] * (self.seq_len - len(sequence)))
            
            inputs = torch.tensor(sequence[:-1], dtype=torch.long)
            targets = torch.tensor(sequence[1:], dtype=torch.long)
            
            # Set target for the answer position
            answer_pos = len(sequence) - 1
            if answer_pos < len(targets):
                targets[answer_pos - 1] = kv_pairs[query_key]
            
            data.append((inputs, targets))
        return data
    
    def _generate_long_infill_task(self):
        """Fill missing parts in long sequences."""
        data = []
        for _ in range(self.num_samples):
            # Generate base pattern
            pattern_len = random.randint(5, 20)
            pattern = torch.randint(1, min(100, self.vocab_size), (pattern_len,))
            
            # Repeat pattern with variations
            full_sequence = []
            num_repeats = self.seq_len // pattern_len
            
            for i in range(num_repeats):
                if i < num_repeats - 1:  # Not the last repeat
                    full_sequence.extend(pattern.tolist())
                else:
                    # Partial pattern for infill task
                    cutoff = random.randint(1, pattern_len - 1)
                    full_sequence.extend(pattern[:cutoff].tolist())
                    
                    # Add mask token for what needs to be filled
                    full_sequence.extend([2] * (pattern_len - cutoff))  # 2 is mask token
            
            # Pad if necessary
            if len(full_sequence) < self.seq_len:
                full_sequence.extend([0] * (self.seq_len - len(full_sequence)))
            elif len(full_sequence) > self.seq_len:
                full_sequence = full_sequence[:self.seq_len]
            
            inputs = torch.tensor(full_sequence[:-1], dtype=torch.long)
            
            # Create targets: replace mask tokens with correct pattern
            targets = torch.tensor(full_sequence[1:], dtype=torch.long)
            mask_positions = (inputs == 2)
            if mask_positions.any():
                # Find what should be filled based on pattern
                pattern_cycle = pattern.repeat((self.seq_len // pattern_len) + 1)[:self.seq_len-1]
                targets[mask_positions] = pattern_cycle[mask_positions]
            
            data.append((inputs, targets))
        return data
    
    def _generate_needle_haystack(self):
        """Needle in haystack - find specific information in long context."""
        data = []
        for _ in range(self.num_samples):
            # Generate haystack (random tokens)
            haystack = torch.randint(100, self.vocab_size, (self.seq_len - 20,))
            
            # Insert needle (special pattern) at random position
            needle = torch.tensor([999, 998, 997])  # Special needle pattern
            needle_pos = random.randint(10, len(haystack) - 10)
            
            # Construct sequence: haystack with needle
            sequence = torch.cat([
                haystack[:needle_pos],
                needle,
                haystack[needle_pos:needle_pos + len(haystack) - len(needle)]
            ])
            
            # Add query at the end: "Find needle position"
            query = torch.tensor([999, 0])  # Query for needle
            sequence = torch.cat([sequence, query])
            
            # Pad to exact length
            if len(sequence) < self.seq_len:
                padding = torch.zeros(self.seq_len - len(sequence), dtype=torch.long)
                sequence = torch.cat([sequence, padding])
            else:
                sequence = sequence[:self.seq_len]
            
            inputs = sequence[:-1]
            targets = sequence[1:]
            
            # Set target as needle position for the query
            targets[-1] = needle_pos
            
            data.append((inputs, targets))
        return data
    
    def _generate_associative_recall(self):
        """Test associative memory - recall associated items."""
        data = []
        for _ in range(self.num_samples):
            # Create associations: A -> B, B -> C, etc.
            associations = {}
            chain_length = random.randint(5, 10)
            
            for i in range(chain_length):
                key = random.randint(10, 99)
                value = random.randint(100, 199)
                associations[key] = value
            
            # Build sequence with associations
            sequence = []
            for key, value in associations.items():
                sequence.extend([key, 3, value, 3])  # 3 is association marker
            
            # Add distractor content
            distractor_len = self.seq_len - len(sequence) - 10
            if distractor_len > 0:
                distractors = torch.randint(200, self.vocab_size, (distractor_len,)).tolist()
                sequence.extend(distractors)
            
            # Add chain query: start -> ? -> ? -> end
            start_key = random.choice(list(associations.keys()))
            sequence.extend([start_key, 4, 0])  # 4 is chain query marker, 0 is placeholder
            
            # Pad
            if len(sequence) < self.seq_len:
                sequence.extend([0] * (self.seq_len - len(sequence)))
            
            inputs = torch.tensor(sequence[:-1], dtype=torch.long)
            targets = torch.tensor(sequence[1:], dtype=torch.long)
            
            # Set target for chain query
            if start_key in associations:
                targets[-1] = associations[start_key]
            
            data.append((inputs, targets))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_toy_dataset(task='copy', num_samples=1000, seq_len=128, vocab_size=1000):
    """
    Creates sophisticated datasets for long-range reasoning.
    
    Available tasks:
    - 'copy': Progressive difficulty copy task
    - 'lookup': Key-value lookup in long context
    - 'long_infill': Pattern completion over long sequences  
    - 'needle_haystack': Find specific info in long context
    - 'associative_recall': Test associative memory chains
    """
    dataset = LongRangeDataset(task, num_samples, seq_len, vocab_size)
    return dataset

def create_curriculum_dataset(tasks=['copy', 'lookup', 'long_infill'], 
                            num_samples_per_task=1000, seq_len=128, vocab_size=1000):
    """Create a curriculum learning dataset with multiple tasks."""
    all_data = []
    
    for task in tasks:
        task_dataset = create_toy_dataset(task, num_samples_per_task, seq_len, vocab_size)
        task_data = [(inp, tgt, task) for inp, tgt in task_dataset]
        all_data.extend(task_data)
    
    # Shuffle for mixed training
    random.shuffle(all_data)
    
    return all_data
