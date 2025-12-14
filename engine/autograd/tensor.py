import numpy as np
from typing import List, Callable, Optional, Any


class Tensor:
    """Tensor with automatic differentiation support"""
    
    def __init__(self, value: np.ndarray, 
                 parents: Optional[List['Tensor']] = None, 
                 backward_fn: Optional[Callable] = None,
                 requires_grad: bool = True):
        self.value = np.asarray(value)
        self.parents = parents or []
        self.backward_fn = backward_fn
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.value) if requires_grad else None
        
        # For computational graph tracking
        self._generation = 0
        if parents:
            self._generation = max(p._generation for p in parents) + 1
    
    def zero_grad(self) -> None:
        """Reset gradient to zero"""
        if self.requires_grad:
            self.grad = np.zeros_like(self.value)
    
    def backward(self, gradient: Optional[np.ndarray] = None) -> None:
        """Compute gradients using backward pass"""
        if not self.requires_grad:
            return
        
        if gradient is None:
            # For scalar output, gradient is 1
            if self.value.size == 1:
                gradient = np.ones_like(self.value)
            else:
                raise ValueError("Gradient must be provided for non-scalar tensors")
        
        # Accumulate gradient
        if self.grad is None:
            self.grad = gradient.copy()
        else:
            self.grad += gradient
        
        # Propagate to parents
        if self.backward_fn is not None and self.parents:
            parent_gradients = self.backward_fn(gradient)
            if parent_gradients is not None:
                for parent, parent_grad in zip(self.parents, parent_gradients):
                    if parent_grad is not None:
                        parent.backward(parent_grad)
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        """Addition operation"""
        other = _ensure_tensor(other)
        
        def backward_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad, grad]
        
        return Tensor(
            self.value + other.value,
            parents=[self, other],
            backward_fn=backward_fn,
            requires_grad=self.requires_grad or other.requires_grad
        )
    
    def __radd__(self, other: 'Tensor') -> 'Tensor':
        """Reverse addition"""
        return self + other
    
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        """Subtraction operation"""
        other = _ensure_tensor(other)
        
        def backward_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad, -grad]
        
        return Tensor(
            self.value - other.value,
            parents=[self, other],
            backward_fn=backward_fn,
            requires_grad=self.requires_grad or other.requires_grad
        )
    
    def __rsub__(self, other: 'Tensor') -> 'Tensor':
        """Reverse subtraction"""
        other = _ensure_tensor(other)
        return other - self
    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        """Multiplication operation"""
        other = _ensure_tensor(other)
        
        def backward_fn(grad: np.ndarray) -> List[np.ndarray]:
            # For element-wise multiplication
            return [grad * other.value, grad * self.value]
        
        return Tensor(
            self.value * other.value,
            parents=[self, other],
            backward_fn=backward_fn,
            requires_grad=self.requires_grad or other.requires_grad
        )
    
    def __rmul__(self, other: 'Tensor') -> 'Tensor':
        """Reverse multiplication"""
        return self * other
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication operation"""
        other = _ensure_tensor(other)
        
        def backward_fn(grad: np.ndarray) -> List[np.ndarray]:
            # For matrix multiplication: C = A @ B
            # dL/dA = dL/dC @ B^T
            # dL/dB = A^T @ dL/dC
            grad_self = grad @ other.value.T
            grad_other = self.value.T @ grad
            return [grad_self, grad_other]
        
        return Tensor(
            self.value @ other.value,
            parents=[self, other],
            backward_fn=backward_fn,
            requires_grad=self.requires_grad or other.requires_grad
        )
    
    def __rmatmul__(self, other: 'Tensor') -> 'Tensor':
        """Reverse matrix multiplication"""
        other = _ensure_tensor(other)
        return other @ self
    
    def __neg__(self) -> 'Tensor':
        """Negation operation"""
        def backward_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [-grad]
        
        return Tensor(
            -self.value,
            parents=[self],
            backward_fn=backward_fn,
            requires_grad=self.requires_grad
        )
    
    def __pow__(self, power: float) -> 'Tensor':
        """Power operation"""
        def backward_fn(grad: np.ndarray) -> List[np.ndarray]:
            return [grad * power * np.power(self.value, power - 1)]
        
        return Tensor(
            np.power(self.value, power),
            parents=[self],
            backward_fn=backward_fn,
            requires_grad=self.requires_grad
        )
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.value.shape}, requires_grad={self.requires_grad})"
    
    def item(self) -> Any:
        """Get scalar value"""
        if self.value.size == 1:
            return self.value.item()
        else:
            raise ValueError("Only scalar tensors have item() method")
    
    def sum(self, axis: Optional[int] = None) -> 'Tensor':
        """Sum operation"""
        def backward_fn(grad: np.ndarray) -> List[np.ndarray]:
            # For sum, gradient is 1 for each element
            if axis is None:
                return [np.ones_like(self.value) * grad]
            else:
                # For axis-specific sum, broadcast gradient
                grad_shape = list(self.value.shape)
                grad_shape[axis] = 1
                return [np.broadcast_to(grad, self.value.shape) * grad]
        
        return Tensor(
            np.sum(self.value, axis=axis),
            parents=[self],
            backward_fn=backward_fn,
            requires_grad=self.requires_grad
        )


def _ensure_tensor(obj: Any) -> Tensor:
    """Convert object to Tensor if needed"""
    if isinstance(obj, Tensor):
        return obj
    else:
        return Tensor(obj, requires_grad=False)


# Convenience functions
def tensor(value: np.ndarray, requires_grad: bool = True) -> Tensor:
    """Create a tensor"""
    return Tensor(value, requires_grad=requires_grad)


def zeros(shape: tuple, requires_grad: bool = True) -> Tensor:
    """Create zero tensor"""
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(shape: tuple, requires_grad: bool = True) -> Tensor:
    """Create ones tensor"""
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(shape: tuple, requires_grad: bool = True) -> Tensor:
    """Create random normal tensor"""
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)
