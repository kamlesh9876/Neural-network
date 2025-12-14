import numpy as np
from typing import Tuple, Optional


class Dropout:
    """Dropout layer"""
    
    def __init__(self, dropout_rate: float = 0.2):
        self.dropout_rate = dropout_rate
        self.mask: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with dropout"""
        if training and self.dropout_rate > 0:
            self.mask = (np.random.rand(*x.shape) < (1 - self.dropout_rate)) / (1 - self.dropout_rate)
            return x * self.mask
        return x
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass"""
        if self.mask is not None:
            return dout * self.mask
        return dout
