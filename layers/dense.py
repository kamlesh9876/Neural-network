import numpy as np
from typing import Tuple


class Dense:
    """Fully connected layer"""
    
    def __init__(self, input_size: int, output_size: int):
        # He initialization
        std = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * std
        self.biases = np.zeros((1, output_size))
        
        # For momentum
        self.momentum_weights = np.zeros_like(self.weights)
        self.momentum_biases = np.zeros_like(self.biases)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: X @ W + b"""
        return np.dot(X, self.weights) + self.biases
    
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get weights and biases"""
        return self.weights, self.biases
    
    def get_momentum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get momentum terms"""
        return self.momentum_weights, self.momentum_biases
