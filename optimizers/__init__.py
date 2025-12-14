import numpy as np
from typing import List, Tuple


class Optimizer:
    """Base optimizer class"""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
    
    def update(self, weights: List[np.ndarray], biases: List[np.ndarray], 
               dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with momentum optimizer"""
    
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.momentum_weights = []
        self.momentum_biases = []
    
    def initialize(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        """Initialize momentum terms"""
        self.momentum_weights = [np.zeros_like(w) for w in weights]
        self.momentum_biases = [np.zeros_like(b) for b in biases]
    
    def update(self, weights: List[np.ndarray], biases: List[np.ndarray], 
               dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        """Update weights using SGD with momentum"""
        if not self.momentum_weights:
            self.initialize(weights, biases)
        
        for i in range(len(weights)):
            # Update momentum terms
            self.momentum_weights[i] = (self.momentum * self.momentum_weights[i] - 
                                        self.learning_rate * dW[i])
            self.momentum_biases[i] = (self.momentum * self.momentum_biases[i] - 
                                       self.learning_rate * db[i])
            
            # Update parameters
            weights[i] += self.momentum_weights[i]
            biases[i] += self.momentum_biases[i]


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.t = 0
    
    def initialize(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        """Initialize first and second moment estimates"""
        self.m_w = [np.zeros_like(w) for w in weights]
        self.v_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_b = [np.zeros_like(b) for b in biases]
        self.t = 0
    
    def update(self, weights: List[np.ndarray], biases: List[np.ndarray], 
               dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        """Update weights using Adam optimizer"""
        if not self.m_w:
            self.initialize(weights, biases)
        
        self.t += 1
        
        for i in range(len(weights)):
            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
            
            # Update biased second raw moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dW[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db[i] ** 2)
            
            # Compute bias-corrected estimates
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.cache_w = []
        self.cache_b = []
    
    def initialize(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        """Initialize cache"""
        self.cache_w = [np.zeros_like(w) for w in weights]
        self.cache_b = [np.zeros_like(b) for b in biases]
    
    def update(self, weights: List[np.ndarray], biases: List[np.ndarray], 
               dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        """Update weights using RMSprop optimizer"""
        if not self.cache_w:
            self.initialize(weights, biases)
        
        for i in range(len(weights)):
            # Update cache
            self.cache_w[i] = self.rho * self.cache_w[i] + (1 - self.rho) * (dW[i] ** 2)
            self.cache_b[i] = self.rho * self.cache_b[i] + (1 - self.rho) * (db[i] ** 2)
            
            # Update parameters
            weights[i] -= self.learning_rate * dW[i] / (np.sqrt(self.cache_w[i]) + self.epsilon)
            biases[i] -= self.learning_rate * db[i] / (np.sqrt(self.cache_b[i]) + self.epsilon)


def get_optimizer(name: str, learning_rate: float = 0.001, **kwargs) -> Optimizer:
    """Get optimizer by name"""
    optimizers = {
        'sgd': SGD(learning_rate=learning_rate, momentum=kwargs.get('momentum', 0.9)),
        'adam': Adam(learning_rate=learning_rate),
        'rmsprop': RMSprop(learning_rate=learning_rate)
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizers[name]