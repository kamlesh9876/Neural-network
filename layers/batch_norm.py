import numpy as np
from typing import Tuple, Optional


class BatchNorm:
    """Batch normalization layer"""
    
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        # Running statistics (for inference)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Cache for backward pass
        self.cache: Optional[Tuple] = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with batch normalization"""
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
            
            # Cache for backward pass
            self.cache = (x, x_norm, batch_mean, batch_var, self.gamma, self.eps)
            
            return out
        else:
            # Inference mode - use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass"""
        if self.cache is None:
            raise RuntimeError("No cache available. Call forward() with training=True first.")
        
        x, x_norm, mean, var, gamma, eps = self.cache
        N, D = dout.shape
        
        # Gradients of batch norm parameters
        dgamma = np.sum(dout * x_norm, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient of input (simplified version)
        dx_norm = dout * gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps) ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=0, keepdims=True)
        dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N
        
        return dx, dgamma, dbeta
    
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get gamma and beta parameters"""
        return self.gamma, self.beta
