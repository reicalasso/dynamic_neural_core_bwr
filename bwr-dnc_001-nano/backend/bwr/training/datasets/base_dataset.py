"""
Base dataset classes for DNC training.
"""

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base class for all DNC datasets."""
    
    def __init__(self):
        super().__init__()
    
    def _generate_data(self):
        """Generate dataset data."""
        raise NotImplementedError("Subclasses must implement _generate_data method")