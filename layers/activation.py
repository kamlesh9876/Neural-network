import numpy as np
from typing import Tuple


class Activation:
    """Base activation function class"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    """ReLU activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class LeakyReLU(Activation):
    """Leaky ReLU activation function"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)


class ELU(Activation):
    """ELU activation function"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha * np.exp(x))


class GELU(Activation):
    """GELU activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        return cdf + x * (1 - cdf ** 2) * np.sqrt(2 / np.pi) * (1 + 0.134145 * x ** 2)


def get_activation(name: str) -> Activation:
    """Get activation function by name"""
    activations = {
        'relu': ReLU(),
        'leaky_relu': LeakyReLU(),
        'elu': ELU(),
        'gelu': GELU()
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activations[name]
