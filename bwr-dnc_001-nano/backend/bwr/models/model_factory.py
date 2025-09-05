"""
Model factory for creating DNC models.
"""

from .basic_dnc import BasicDNC
from .hierarchical_dnc import HierarchicalDNC

class ModelFactory:
    """Factory for creating DNC models."""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Create a DNC model.
        
        Args:
            model_type: Type of model to create ("basic" or "hierarchical")
            **kwargs: Model parameters
            
        Returns:
            DNC model instance
        """
        if model_type == "basic":
            return BasicDNC(**kwargs)
        elif model_type == "hierarchical":
            return HierarchicalDNC(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")