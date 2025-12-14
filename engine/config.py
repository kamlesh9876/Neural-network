from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for training the neural network"""
    input_size: int = 784
    hidden_sizes: List[int] = None
    output_size: int = 10
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 100
    momentum: float = 0.9
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = 'relu'
    optimizer: str = 'adam'
    weight_decay: float = 1e-4
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64]
