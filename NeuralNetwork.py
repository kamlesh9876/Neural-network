import numpy as np
import matplotlib.pyplot as plt
import os
import json
import signal
import sys
import datetime
import argparse
import math
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

import matplotlib
# Use TkAgg backend for interactive plotting
matplotlib.use('TkAgg')
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

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
    
    @classmethod
    def from_args(cls, args):
        return cls(
            input_size=args.input_size,
            hidden_sizes=list(map(int, args.hidden_sizes.split(','))) if args.hidden_sizes else None,
            output_size=args.output_size,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            momentum=args.momentum,
            dropout_rate=args.dropout_rate,
            use_batch_norm=not args.no_batch_norm,
            activation=args.activation,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay
        )


class NeuralNetwork:
    def __init__(self, config: TrainingConfig):
        """
        Initialize the neural network with configurable architecture and training parameters.
        
        Args:
            config: TrainingConfig object containing model and training parameters
        """
        self.config = config
        
        # Initialize model parameters
        self.weights = []
        self.biases = []
        self.momentum_weights = []
        self.momentum_biases = []
        
        # Initialize weights and biases for each layer
        layer_sizes = [config.input_size] + config.hidden_sizes + [config.output_size]
        
        # Initialize weights using He initialization
        for i in range(len(layer_sizes) - 1):
            # He initialization with ReLU
            std = np.sqrt(2.0 / layer_sizes[i])
            weights = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * std
            bias = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(weights)
            self.biases.append(bias)
            
            # Initialize momentum terms
            self.momentum_weights.append(np.zeros_like(weights))
            self.momentum_biases.append(np.zeros_like(bias))
        
        # Store training history and metadata
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.learning_rates = []
        
        # Training metadata
        self.epoch_count = 0
        self.save_interval = 5  # Save model every 5 epochs
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        self.training_start_time = None
        
        # Create visualization and log directories
        self.viz_dir = os.path.join(self.checkpoint_dir, 'training_visualization')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set up training log file
        self.training_log = os.path.join(self.checkpoint_dir, 'training_log.txt')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training tracking
        self.last_save_time = None
        self.last_plot_time = time.time()
        self.plot_interval = 30  # Update plot every 30 seconds
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_weights = None
        self.best_biases = None
        
        # Enable interactive mode for plots
        plt.ion()
        
    def plot_training_history(self, show=True, save_path=None, interactive=True):
        """Plot training and validation metrics with interactive features."""
        if not hasattr(self, 'loss_history') or not self.loss_history:
            print("No training history to plot.")
            return
            
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Smoothing function
        def smooth_curve(points, factor=0.9):
            return np.convolve(points, np.ones(factor) / factor, mode='valid')
        
        epochs = range(1, len(self.loss_history) + 1)
        
        # Plot Loss
        smooth_loss = smooth_curve(self.loss_history)
        ax1.plot(epochs[:len(smooth_loss)], smooth_loss, 'b-', label='Training Loss')
        
        if hasattr(self, 'val_loss_history') and self.val_loss_history:
            smooth_val = smooth_curve(self.val_loss_history)
            ax1.plot(epochs[:len(smooth_val)], smooth_val, 'r-', label='Validation Loss')
        
        ax1.set_title('Training & Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot Accuracy
        if hasattr(self, 'accuracy_history') and self.accuracy_history:
            smooth_acc = smooth_curve(self.accuracy_history)
            ax2.plot(epochs[:len(smooth_acc)], smooth_acc, 'b-', label='Training Accuracy')
            
            if hasattr(self, 'val_accuracy_history') and self.val_accuracy_history:
                smooth_val = smooth_curve(self.val_accuracy_history)
                ax2.plot(epochs[:len(smooth_val)], smooth_val, 'r-', label='Validation Accuracy')
        
        ax2.set_title('Training & Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Add training info
        if hasattr(self, 'training_start_time') and self.training_start_time:
            elapsed = time.time() - self.training_start_time
            hours, rem = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(rem, 60)
            plt.suptitle(f"Epochs: {len(self.loss_history)} | Time: {hours:02d}:{minutes:02d}:{seconds:02d}", 
                        fontsize=12, y=0.99)
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f"Error saving plot: {e}")
        
        # Show or close
        if show:
            if interactive:
                plt.show(block=False)
                plt.pause(0.1)
            else:
                plt.show()
        else:
            plt.close()
        # Initialize training start time
        self.training_start_time = None
        
    def _get_activation(self, name):
        """Get activation function by name"""
        if name == 'relu':
            return self._relu, self._relu_derivative
        elif name == 'leaky_relu':
            return self._leaky_relu, self._leaky_relu_derivative
        elif name == 'elu':
            return self._elu, self._elu_derivative
        elif name == 'gelu':
            return self._gelu, self._gelu_derivative
        else:
            raise ValueError(f"Unknown activation function: {name}")
    
    # Activation functions
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def _leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def _elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def _elu_derivative(self, x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    def _gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def _gelu_derivative(self, x):
        # Approximate derivative of GELU
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        return cdf + x * (1 - cdf ** 2) * np.sqrt(2 / np.pi) * (1 + 0.134145 * x ** 2)
    
    def _batch_norm_forward(self, x, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9, training=True):
        """Batch normalization forward pass"""
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            if moving_mean is not None and moving_var is not None:
                moving_mean = momentum * moving_mean + (1 - momentum) * batch_mean
                moving_var = momentum * moving_var + (1 - momentum) * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)
            
            # Scale and shift
            out = gamma * x_norm + beta
            
            # Cache for backward pass
            cache = (x, x_norm, batch_mean, batch_var, gamma, eps)
            
            return out, cache, moving_mean, moving_var
        else:
            # Inference mode - use moving statistics
            x_norm = (x - moving_mean) / np.sqrt(moving_var + eps)
            return gamma * x_norm + beta, None, moving_mean, moving_var
    
    def _batch_norm_backward(self, dout, cache):
        """Batch normalization backward pass"""
        x, x_norm, mean, var, gamma, eps = cache
        
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
    
    def _dropout_forward(self, x, p_dropout, training=True):
        """Dropout forward pass"""
        if training and p_dropout > 0:
            mask = (np.random.rand(*x.shape) < (1 - p_dropout)) / (1 - p_dropout)
            return x * mask, mask
        return x, None
    
    def _dropout_backward(self, dout, mask):
        """Dropout backward pass"""
        if mask is not None:
            return dout * mask
        return dout
    
    def forward(self, X, training=True):
        """
        Forward pass through the network.
        
        Args:
            X: Input data (n_samples, input_size)
            training: If True, store intermediate values for backpropagation
            
        Returns:
            Network output and cache for backward pass
        """
        cache = {'A': [X]}
        A = X
        
        # Forward pass through all hidden layers
        for i in range(len(self.weights) - 1):
            # Linear transformation
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            
            # Batch normalization
            if self.config.use_batch_norm:
                if not hasattr(self, f'bn_gamma_{i}'):
                    # Initialize batch norm parameters if they don't exist
                    setattr(self, f'bn_gamma_{i}', np.ones((1, Z.shape[1])))
                    setattr(self, f'bn_beta_{i}', np.zeros((1, Z.shape[1])))
                    setattr(self, f'running_mean_{i}', np.zeros((1, Z.shape[1])))
                    setattr(self, f'running_var_{i}', np.ones((1, Z.shape[1])))
                
                gamma = getattr(self, f'bn_gamma_{i}')
                beta = getattr(self, f'bn_beta_{i}')
                running_mean = getattr(self, f'running_mean_{i}')
                running_var = getattr(self, f'running_var_{i}')
                
                Z_norm, bn_cache, new_mean, new_var = self._batch_norm_forward(
                    Z, gamma, beta, running_mean, running_var, training=training
                )
                
                if training:
                    setattr(self, f'running_mean_{i}', new_mean)
                    setattr(self, f'running_var_{i}', new_var)
                    cache[f'bn_{i}'] = bn_cache
                
                Z = Z_norm
            
            # Activation
            activation, _ = self._get_activation(self.config.activation)
            A = activation(Z)
            
            # Store activations for next layer
            cache[f'A_{i+1}'] = A
            cache[f'Z_{i}'] = Z
        
        # Output layer (no activation, softmax will be applied in the loss function)
        Z_out = np.dot(A, self.weights[-1]) + self.biases[-1]
        cache['Z_out'] = Z_out
        
        # Apply softmax for classification
        exp_scores = np.exp(Z_out - np.max(Z_out, axis=1, keepdims=True))
        output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return output, cache
    
    def _batch_norm_backward(self, dout, cache):
        """Batch normalization backward pass"""
        x, x_norm, mean, var, gamma, eps = cache
        
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
    
    def save_model(self, filename=None):
        """Save model weights and training history to disk"""
        if filename is None:
            filename = f'model_epoch_{self.epoch_count:04d}.npz'
        
        model_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save model weights and metadata
        model_data = {}
        
        # Save weights and biases with unique keys
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            model_data[f'weight_{i}'] = w
            model_data[f'bias_{i}'] = b
        
        # Add training history
        model_data['loss_history'] = np.array(self.loss_history, dtype=np.float32)
        model_data['accuracy_history'] = np.array(self.accuracy_history, dtype=np.float32)
        model_data['epoch'] = np.array([self.epoch_count], dtype=np.int32)
        
        # Add batch norm parameters if used
        if self.config.use_batch_norm:
            for i in range(len(self.weights) - 1):
                model_data[f'bn_gamma_{i}'] = getattr(self, f'bn_gamma_{i}', np.array([]))
                model_data[f'bn_beta_{i}'] = getattr(self, f'bn_beta_{i}', np.array([]))
                model_data[f'running_mean_{i}'] = getattr(self, f'running_mean_{i}', np.array([]))
                model_data[f'running_var_{i}'] = getattr(self, f'running_var_{i}', np.array([]))
        
        # Save optimizer state if it exists
        if self.config.optimizer == 'adam' and hasattr(self, 'm_w'):
            for i in range(len(self.weights)):
                model_data[f'adam_m_w_{i}'] = self.m_w[i]
                model_data[f'adam_v_w_{i}'] = self.v_w[i]
                model_data[f'adam_m_b_{i}'] = self.m_b[i]
                model_data[f'adam_v_b_{i}'] = self.v_b[i]
            model_data['adam_t'] = np.array([self.t])
        elif self.config.optimizer == 'rmsprop' and hasattr(self, 'cache_w'):
            for i in range(len(self.weights)):
                model_data[f'rms_w_{i}'] = self.cache_w[i]
                model_data[f'rms_b_{i}'] = self.cache_b[i]
        
        # Save the model data
        np.savez_compressed(model_path, **model_data)
        
        # Save training history as JSON with proper NumPy type conversion
        history = {
            'epoch': int(self.epoch_count),
            'loss': [float(loss) for loss in self.loss_history],
            'accuracy': [float(acc) for acc in self.accuracy_history],
            'timestamp': float(time.time()),
            'training_time': float(time.time() - self.training_start_time) if self.training_start_time else 0.0,
            'config': {
                'learning_rate': float(self.config.learning_rate),
                'batch_size': int(self.config.batch_size),
                'epochs': int(self.config.epochs),
                'optimizer': str(self.config.optimizer),
                'use_batch_norm': bool(self.config.use_batch_norm),
                'dropout_rate': float(self.config.dropout_rate)
            }
        }
        
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else int(x))
        
        # Update last save time
        self.last_save_time = time.time()
        
        # Log the save event
        self._log_training(f"Model saved to {model_path}")
        return model_path
    
    def compute_loss(self, y_true, y_pred, eps=1e-15):
        """
        Compute cross-entropy loss with L2 regularization.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            eps: Small constant for numerical stability
            
        Returns:
            Cross-entropy loss with L2 regularization
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Cross-entropy loss
        n_samples = y_true.shape[0]
        cross_entropy = -np.sum(y_true * np.log(y_pred)) / n_samples
        
        # L2 regularization
        l2_reg = 0
        if self.config.weight_decay > 0:
            for w in self.weights:
                l2_reg += np.sum(w ** 2)
            l2_reg = (self.config.weight_decay / 2) * l2_reg / n_samples
        
        return cross_entropy + l2_reg
    
    def predict(self, X, training=False):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data (n_samples, n_features)
            training: If True, applies dropout and batch norm in training mode
            
        Returns:
            Predicted class probabilities
        """
        # Forward pass
        output, _ = self.forward(X, training=training)
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probabilities
    
    def backward(self, X, y, output, cache, learning_rate):
        """
        Perform backward propagation and update weights.
        
        Args:
            X: Input data (batch_size, n_features)
            y: True labels (one-hot encoded, batch_size, n_classes)
            output: Network output (batch_size, n_classes)
            cache: Dictionary containing intermediate values from forward pass
            learning_rate: Current learning rate
        """
        n_samples = X.shape[0]
        
        # Calculate gradient of loss with respect to output
        dZ = output - y  # Derivative of cross-entropy loss with softmax
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Backpropagate through the network
        for l in reversed(range(len(self.weights))):
            # Get activations from cache
            A_prev = cache[f'A_{l}'] if l > 0 else X
            
            # Compute gradients for weights and biases
            dW[l] = np.dot(A_prev.T, dZ) / n_samples
            db[l] = np.sum(dZ, axis=0, keepdims=True) / n_samples
            
            # Add L2 regularization if weight decay is enabled
            if self.config.weight_decay > 0:
                dW[l] += self.config.weight_decay * self.weights[l] / n_samples
            
            # If not the first layer, compute gradient for previous layer
            if l > 0:
                # Get activation derivative
                if self.config.activation == 'relu':
                    dA = self._relu_derivative(cache[f'Z_{l-1}'])
                elif self.config.activation == 'leaky_relu':
                    dA = self._leaky_relu_derivative(cache[f'Z_{l-1}'])
                elif self.config.activation == 'elu':
                    dA = self._elu_derivative(cache[f'Z_{l-1}'])
                elif self.config.activation == 'gelu':
                    dA = self._gelu_derivative(cache[f'Z_{l-1}'])
                else:
                    raise ValueError(f"Unsupported activation: {self.config.activation}")
                
                # Compute gradient for previous layer
                dZ = np.dot(dZ, self.weights[l].T) * dA
                
                # Apply dropout if enabled
                if self.config.dropout_rate > 0 and f'dropout_{l-1}' in cache:
                    dZ *= cache[f'dropout_{l-1}']
        
        # Update weights and biases using the optimizer
        if self.config.optimizer == 'sgd':
            # SGD with momentum
            for l in range(len(self.weights)):
                # Update momentum terms
                self.momentum_weights[l] = (self.config.momentum * self.momentum_weights[l] - 
                                          learning_rate * dW[l])
                self.momentum_biases[l] = (self.config.momentum * self.momentum_biases[l] - 
                                         learning_rate * db[l])
                
                # Update parameters
                self.weights[l] += self.momentum_weights[l]
                self.biases[l] += self.momentum_biases[l]
                
        elif self.config.optimizer == 'adam':
            # Adam optimizer
            if not hasattr(self, 'm_w'):
                # Initialize first and second moment estimates
                self.m_w = [np.zeros_like(w) for w in self.weights]
                self.v_w = [np.zeros_like(w) for w in self.weights]
                self.m_b = [np.zeros_like(b) for b in self.biases]
                self.v_b = [np.zeros_like(b) for b in self.biases]
                self.t = 0
            
            self.t += 1
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
            
            for l in range(len(self.weights)):
                # Update biased first moment estimate
                self.m_w[l] = beta1 * self.m_w[l] + (1 - beta1) * dW[l]
                self.m_b[l] = beta1 * self.m_b[l] + (1 - beta1) * db[l]
                
                # Update biased second raw moment estimate
                self.v_w[l] = beta2 * self.v_w[l] + (1 - beta2) * (dW[l] ** 2)
                self.v_b[l] = beta2 * self.v_b[l] + (1 - beta2) * (db[l] ** 2)
                
                # Compute bias-corrected first moment estimate
                m_w_hat = self.m_w[l] / (1 - beta1 ** self.t)
                m_b_hat = self.m_b[l] / (1 - beta1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_w_hat = self.v_w[l] / (1 - beta2 ** self.t)
                v_b_hat = self.v_b[l] / (1 - beta2 ** self.t)
                
                # Update parameters
                self.weights[l] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.biases[l] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        elif self.config.optimizer == 'rmsprop':
            # RMSprop optimizer
            if not hasattr(self, 'cache_w'):
                self.cache_w = [np.zeros_like(w) for w in self.weights]
                self.cache_b = [np.zeros_like(b) for b in self.biases]
            
            rho = 0.9
            epsilon = 1e-8
            
            for l in range(len(self.weights)):
                # Update cache
                self.cache_w[l] = rho * self.cache_w[l] + (1 - rho) * (dW[l] ** 2)
                self.cache_b[l] = rho * self.cache_b[l] + (1 - rho) * (db[l] ** 2)
                
                # Update parameters
                self.weights[l] -= learning_rate * dW[l] / (np.sqrt(self.cache_w[l]) + epsilon)
                self.biases[l] -= learning_rate * db[l] / (np.sqrt(self.cache_b[l]) + epsilon)
        
        else:
            # Vanilla SGD (without momentum)
            for l in range(len(self.weights)):
                self.weights[l] -= learning_rate * dW[l]
                self.biases[l] -= learning_rate * db[l]
    
    def compute_accuracy(self, y_true, y_pred):
        """
        Compute classification accuracy.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities or logits
            
        Returns:
            Accuracy in percentage
        """
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Convert from probabilities to class indices
            y_pred = np.argmax(y_pred, axis=1)
        
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            # Convert from one-hot to class indices
            y_true = np.argmax(y_true, axis=1)
        
        accuracy = np.mean(y_pred == y_true) * 100.0
        return accuracy
    
    def _format_time(self, seconds):
        """Convert seconds to human readable format (HH:MM:SS or D days, HH:MM:SS if > 1 day)"""
        seconds = int(seconds)
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _log_training(self, message, is_progress=False):
        """Log training progress to file and console"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Print to console (only progress messages if it's a progress update)
        if not is_progress or self.epoch_count % 10 == 0:  # Show progress every 10 epochs
            print(log_message)
        
        # Always write to log file
        with open(self.training_log, 'a') as f:
            f.write(log_message + '\n')
    
    def load_model(self, filename):
        """Load model weights, optimizer state, and history from disk"""
        data = np.load(filename, allow_pickle=True)
        
        # Clear existing weights and biases
        self.weights = []
        self.biases = []
        
        # Load weights and biases for each layer
        i = 0
        while f'weight_{i}' in data:
            self.weights.append(data[f'weight_{i}'].copy())
            self.biases.append(data[f'bias_{i}'].copy())
            i += 1
        
        # Load batch normalization parameters if they exist
        if self.config.use_batch_norm:
            for j in range(len(self.weights) - 1):
                if f'bn_gamma_{j}' in data:
                    setattr(self, f'bn_gamma_{j}', data[f'bn_gamma_{j}'].copy())
                if f'bn_beta_{j}' in data:
                    setattr(self, f'bn_beta_{j}', data[f'bn_beta_{j}'].copy())
                if f'running_mean_{j}' in data:
                    setattr(self, f'running_mean_{j}', data[f'running_mean_{j}'].copy())
                if f'running_var_{j}' in data:
                    setattr(self, f'running_var_{j}', data[f'running_var_{j}'].copy())
        
        # Load optimizer state if it exists
        if self.config.optimizer == 'adam' and f'adam_m_w_0' in data:
            self.m_w = [data[f'adam_m_w_{i}'].copy() for i in range(len(self.weights))]
            self.v_w = [data[f'adam_v_w_{i}'].copy() for i in range(len(self.weights))]
            self.m_b = [data[f'adam_m_b_{i}'].copy() for i in range(len(self.biases))]
            self.v_b = [data[f'adam_v_b_{i}'].copy() for i in range(len(self.biases))]
            self.t = int(data['adam_t'].item()) if 'adam_t' in data else 1
        elif self.config.optimizer == 'rmsprop' and f'rms_w_0' in data:
            self.cache_w = [data[f'rms_w_{i}'].copy() for i in range(len(self.weights))]
            self.cache_b = [data[f'rms_b_{i}'].copy() for i in range(len(self.biases))]
        
        # Load training history
        if 'loss_history' in data:
            self.loss_history = [float(loss) for loss in data['loss_history']]
        if 'accuracy_history' in data:
            self.accuracy_history = [float(acc) for acc in data['accuracy_history']]
        if 'epoch' in data:
            epoch_data = data['epoch']
            if hasattr(epoch_data, 'item'):  # If it's a numpy array/scalar
                self.epoch_count = int(epoch_data.item())
            else:  # If it's already a Python scalar
                self.epoch_count = int(epoch_data)
        
        print(f"Model loaded from {filename}, previous training: {self.epoch_count} epochs")
        print(f"Current learning rate: {self.config.learning_rate}")
        if hasattr(self, 'm_w'):
            print(f"Optimizer: Adam, t={getattr(self, 't', 0)}")
        elif hasattr(self, 'cache_w'):
            print("Optimizer: RMSprop")
        else:
            print("Optimizer: SGD with Momentum")
            
        return self
    
    def plot_training_history(self, show=True, save_path=None):
        "''Plot training loss and accuracy with enhanced visualization"""
        if not self.loss_history:
            print("No training history to plot.")
            return
            
        plt.figure(figsize=(15, 5))
        
        # Plot Loss
        plt.subplot(1, 3, 1)
        plt.plot(self.loss_history, 'b-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(self.accuracy_history, 'g-', label='Training Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot Learning Rate (if available)
        if hasattr(self, 'learning_rate_history') and self.learning_rate_history:
            plt.subplot(1, 3, 3)
            plt.plot(self.learning_rate_history, 'r-', label='Learning Rate')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Rate')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        if save_path is None:
            save_path = os.path.join(self.viz_dir, f'training_history_{int(time.time())}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
    def save_sample_images(self, X, y, num_samples=5, epoch=0):
        """Save sample training images with predictions"""
        if not hasattr(self, 'viz_dir'):
            return None
            
        try:
            # Make predictions
            y_pred = self.predict(X[:num_samples])
            y_pred_labels = np.argmax(y_pred, axis=1)
            
            # If y is one-hot encoded, convert to labels
            if len(y.shape) > 1:
                y_true = np.argmax(y[:num_samples], axis=1)
            else:
                y_true = y[:num_samples]
            
            # Create figure
            plt.figure(figsize=(15, 3))
            
            for i in range(num_samples):
                plt.subplot(1, num_samples, i + 1)
                plt.imshow(X[i].reshape(28, 28), cmap='gray')
                
                # Green for correct prediction, red for incorrect
                color = 'green' if y_pred_labels[i] == y_true[i] else 'red'
                plt.title(f'Pred: {y_pred_labels[i]}\nTrue: {y_true[i]}', color=color)
                plt.axis('off')
            
            plt.tight_layout()
            
            # Create viz directory if it doesn't exist
            os.makedirs(self.viz_dir, exist_ok=True)
            
            # Save the figure with appropriate filename based on epoch type
            if isinstance(epoch, int):
                save_path = os.path.join(self.viz_dir, f'samples_epoch_{epoch:04d}.png')
            else:
                save_path = os.path.join(self.viz_dir, f'samples_{epoch}.png')
                
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"Error saving sample images: {str(e)}")
            return None

    def train(self, X, y, epochs=10, batch_size=64, verbose=True, log_interval=1):
        """
        Train the neural network with mini-batch gradient descent and learning rate decay.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples, n_outputs)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print training progress
            log_interval: Print progress every N epochs
        """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Learning rate scheduling
        initial_lr = self.config.learning_rate
        current_lr = initial_lr
        min_lr = 1e-5
        lr_decay_factor = 0.95
        lr_decay_patience = 2
        lr_decay_counter = 0
        
        # Initialize learning rate history
        if not hasattr(self, 'learning_rate_history'):
            self.learning_rate_history = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            # Shuffle data for this epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Learning rate scheduling
            current_lr = max(min_lr, initial_lr * (lr_decay_factor ** (epoch // lr_decay_patience)))
            self.learning_rate_history.append(current_lr)
            
            for i in range(0, n_samples, batch_size):
                # Get mini-batch
                end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                # Forward pass
                output, a1 = self.forward(X_batch)
                
                # Compute metrics
                batch_loss = self.compute_loss(y_batch, output)
                batch_accuracy = self.compute_accuracy(y_batch, output)
                
                # Update running totals
                batch_size_actual = end - i
                epoch_loss += batch_loss * batch_size_actual
                epoch_accuracy += batch_accuracy * batch_size_actual
                
                # Backward pass and update weights
                self.backward(X_batch, y_batch, output, a1, current_lr)
                
                # Check for keyboard interrupt
                if time.time() - start_time > 1:  # Check every second
                    if signal.getsignal(signal.SIGINT) != signal.SIG_DFL:
                        return
            
            # Calculate epoch metrics
            epoch_loss /= n_samples
            epoch_accuracy /= n_samples
            
            # Store history
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)
            
            # Save sample images every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_sample_images(X, y, epoch=epoch + 1)
            
            # Early stopping check
            if epoch_loss < best_loss - 1e-4:  # Significant improvement
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch > 5:  # Give it at least 5 epochs
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1} - No improvement in {patience} epochs")
                    break
            
            # Print training progress
            if verbose and ((epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1):
                time_elapsed = time.time() - start_time
                epoch_time = time.time() - epoch_start_time
                eta = (time_elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch > 0 else 0
                
                # Format the progress message
                progress_msg = (
                    f"Epoch {epoch + 1:4d}/{epochs} | "
                    f"Loss: {epoch_loss:.6f} | "
                    f"Acc: {epoch_accuracy:6.2f}% | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:5.1f}s | "
                    f"Elapsed: {time_elapsed//60:.0f}m {time_elapsed%60:02.0f}s"
                )
                
                if epoch > 0:
                    progress_msg += f" | ETA: {eta//60:.0f}m {eta%60:02.0f}s"
                
                print(progress_msg)
                
                # Update plot periodically
                current_time = time.time()
                if current_time - self.last_plot_time >= self.plot_interval:
                    self.plot_training_history(show=False)
                    self.last_plot_time = current_time
        
        print(f"\nTraining completed in {time.time() - start_time:.1f} seconds")
        print(f"Final loss: {self.loss_history[-1]:.6f}")
        print(f"Final accuracy: {self.accuracy_history[-1]:.2f}%")
        
        # Plot training history
        self.plot_training_history()
        
        return self.accuracy_history[-1] if self.accuracy_history else 0.0
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST')
    
    # Model architecture
    parser.add_argument('--input-size', type=int, default=784, help='Input size (default: 784 for MNIST)')
    parser.add_argument('--hidden-sizes', type=str, default='128,64', 
                       help='Comma-separated list of hidden layer sizes (default: 128,64)')
    parser.add_argument('--output-size', type=int, default=10, help='Output size (default: 10 for MNIST)')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
    parser.add_argument('--dropout-rate', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    
    # Model options
    parser.add_argument('--no-batch-norm', action='store_true', help='Disable batch normalization')
    parser.add_argument('--activation', type=str, default='relu', 
                       choices=['relu', 'leaky_relu', 'elu', 'gelu'], help='Activation function (default: relu)')
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['sgd', 'adam', 'rmsprop'], help='Optimizer (default: adam)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (L2 penalty) (default: 1e-4)')
    
    # Training options
    parser.add_argument('--validate', action='store_true', help='Use validation split')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio (default: 0.1)')
    parser.add_argument('--early-stopping', type=int, default=10, 
                       help='Patience for early stopping (default: 10, 0 to disable)')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    return parser.parse_args()


def train_continuously():
    # Parse command line arguments
    args = parse_args()
    config = TrainingConfig.from_args(args)
    
    # Set epochs to a very large number for continuous training
    config.epochs = 1000000  # Effectively infinite
    
    # Load MNIST data
    print("Loading MNIST data...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    # Flatten images
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat = X_test.reshape(-1, 784)
    
    print("Data loaded and preprocessed.")
    
    # Create network
    nn = NeuralNetwork(config)
    
    # Initialize training start time and visualization
    nn.training_start_time = time.time()
    nn.last_save_time = nn.training_start_time
    
    # Initialize plot with interactive mode
    plt.ioff()  # Turn off interactive mode temporarily
    plt.close('all')  # Close any existing plots
    
    # Create figure and axes with a specific figure number
    fig = plt.figure(num='Neural Network Training', figsize=(18, 7))
    gs = fig.add_gridspec(1, 2)
    
    # Create the axes using the gridspec
    ax1 = fig.add_subplot(gs[0, 0])  # Left subplot for accuracy
    ax2 = fig.add_subplot(gs[0, 1])  # Right subplot for loss
    
    plt.subplots_adjust(bottom=0.2)  # Make room for info text
    
    # Initialize plot data
    epochs_plot = []
    train_acc_plot = []
    test_acc_plot = []
    loss_plot = []
    
    # Create empty lines for the plot
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Training Accuracy')
    line2, = ax1.plot([], [], 'r-', linewidth=2, label='Test Accuracy')
    line3, = ax2.plot([], [], 'g-', linewidth=2, label='Training Loss')
    
    # Configure plot appearance
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel('Epoch')
    
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.set_ylim(0, 100)  # Set fixed y-limits for accuracy (0-100%)
    
    ax2.set_title('Training Loss')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_ylim(0, 3.0)  # Initial y-limit for loss, will auto-scale
    
    plt.tight_layout()
    plt.ion()  # Turn on interactive mode
    plt.show(block=False)
    plt.pause(0.1)  # Allow the plot to update
    
    # Add a text box for additional info
    info_text = fig.text(0.5, 0.05, 'Initializing training...', 
                        ha='center', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7))
    
    # Initial draw
    plt.draw()
    plt.pause(0.1)
    
    # Create necessary directories
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for existing checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.npz')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        nn._log_training(f"Found and loaded existing checkpoint: {checkpoint_path}")
        nn.load_model(checkpoint_path)
    
    # Handle keyboard interrupt for graceful shutdown
    def signal_handler(sig, frame):
        training_time = time.time() - nn.training_start_time
        nn._log_training("\n" + "="*70)
        nn._log_training("Training Summary")
        nn._log_training("="*70)
        nn._log_training(f"Total training time: {nn._format_time(training_time)}")
        nn._log_training(f"Total epochs completed: {nn.epoch_count}")
        if hasattr(nn, 'accuracy_history') and nn.accuracy_history:
            nn._log_training(f"Best accuracy: {max(nn.accuracy_history):.2f}%")
            nn._log_training(f"Final accuracy: {nn.accuracy_history[-1]:.2f}%")

        # Save final model and visualizations
        nn._log_training("\nSaving final model and visualizations...")
        final_model_path = os.path.join(checkpoint_dir, 'final_model.npz')
        nn.save_model(final_model_path)

        # Save final training visualization
        plot_path = os.path.join(checkpoint_dir, 'training_history_final.png')
        try:
            if hasattr(nn, 'loss_history') and nn.loss_history:
                nn.plot_training_history(save_path=plot_path)
        except Exception as e:
            nn._log_training(f"Warning: Could not save final plot: {str(e)}")

        # Save sample predictions
        if hasattr(nn, 'viz_dir'):
            try:
                sample_path = os.path.join(nn.viz_dir, 'final_predictions.png')
                nn.visualize_predictions(save_path=sample_path)
                nn._log_training(f"Sample predictions: {os.path.abspath(sample_path)}")
            except Exception as e:
                nn._log_training(f"Warning: Could not save sample predictions: {str(e)}")

        nn._log_training(f"\nModel saved to: {os.path.abspath(final_model_path)}")
        nn._log_training(f"Training history plot: {os.path.abspath(plot_path)}")
        nn._log_training("\nTraining completed successfully!")

        # Close the plot window
        plt.close('all')

        # Exit the program
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Print training header
    nn._log_training("\n" + "="*70)
    nn._log_training("Starting Neural Network Training")
    nn._log_training("="*70)
    nn._log_training(f"Training samples: {len(X_train_flat):,}")
    nn._log_training(f"Test samples: {len(X_test_flat):,}")
    nn._log_training(f"Network architecture: 784-{'-'.join(str(s) for s in nn.config.hidden_sizes)}-10")
    nn._log_training(f"Learning rate: {nn.config.learning_rate}, Optimizer: {nn.config.optimizer}")
    if nn.config.optimizer == 'sgd':
        nn._log_training(f"Momentum: {nn.config.momentum}")
    nn._log_training(f"Batch size: {nn.config.batch_size}, Dropout: {nn.config.dropout_rate}")
    nn._log_training(f"Batch norm: {'Enabled' if nn.config.use_batch_norm else 'Disabled'}")
    nn._log_training(f"Checkpoint directory: {os.path.abspath(checkpoint_dir)}")
    nn._log_training("-"*70)
    nn._log_training("Epoch   | Train Acc | Test Acc  | Time/epoch | Elapsed   | Next Save")
    nn._log_training("-"*70)
    
    # Plot initialization is already done above, remove duplicate
    
    try:
        epoch = nn.epoch_count
        last_save_epoch = epoch
        best_accuracy = 0.0
        
        print("\nStarting continuous training. Press Ctrl+C to stop...")
        
        # Main training loop - runs until KeyboardInterrupt
        while True:
            epoch_start_time = time.time()
            
            # Train for one epoch
            nn.train(X_train_flat, y_train_cat, epochs=1, verbose=False)
            
            # Update epoch count
            epoch += 1
            nn.epoch_count = epoch
            
            # Calculate metrics on a subset for speed
            sample_size = min(1000, len(X_test_flat))
            sample_indices = np.random.choice(len(X_test_flat), sample_size, replace=False)
            
            # Calculate training accuracy
            train_output = nn.predict(X_train_flat[sample_indices])
            train_acc = nn.compute_accuracy(y_train_cat[sample_indices], train_output)
            
            test_output = nn.predict(X_test_flat[sample_indices])
            test_acc = nn.compute_accuracy(y_test_cat[sample_indices], test_output)
            
            # Update history and track best accuracy
            if not hasattr(nn, 'accuracy_history'):
                nn.accuracy_history = []
            nn.accuracy_history.append(test_acc)
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                is_best = " (Best!)"
            else:
                is_best = ""
            
            # Calculate times
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - nn.training_start_time
            next_save = max(0, 60 - (time.time() - nn.last_save_time))
            
            # Format progress line
            progress_line = (
                f"{epoch:6d} | "
                f"{train_acc:7.2f}% | "
                f"{test_acc:7.2f}%{is_best:<8} | "
                f"{epoch_time:5.1f}s     | "
                f"{nn._format_time(elapsed_time)} | "
                f"{int(next_save)}s"
            )
            
            # Log progress
            nn._log_training(progress_line, is_progress=True)
            
            # Update plot data
            epochs_plot.append(epoch)
            train_acc_plot.append(train_acc)
            test_acc_plot.append(test_acc)
            
            # Get the latest loss if available
            if hasattr(nn, 'loss_history') and nn.loss_history:
                loss_plot.append(nn.loss_history[-1])
            
            # Update the plot
            if epochs_plot and len(epochs_plot) > 1:
                # Update plot lines
                line1.set_data(epochs_plot, train_acc_plot)
                line2.set_data(epochs_plot, test_acc_plot)
                
                if loss_plot:
                    line3.set_data(range(1, len(loss_plot) + 1), loss_plot)
                
                # Adjust axes with padding
                for ax in [ax1, ax2]:
                    ax.relim()
                    ax.autoscale_view()
                    y_min, y_max = ax.get_ylim()
                    padding = (y_max - y_min) * 0.05
                    ax.set_ylim(y_min - padding, y_max + padding)
                
                # Update info text with more details
                if loss_plot:  # Check if loss_plot is not empty
                    info_text.set_text(
                        f"Epoch: {epoch:4d} | "
                        f"Train: {train_acc:5.2f}% | "
                        f"Test: {test_acc:5.2f}% | "
                        f"Loss: {loss_plot[-1]:.4f} | "
                        f"Time: {nn._format_time(elapsed_time)} | "
                        f"Press Ctrl+C to stop"
                    )
                else:
                    info_text.set_text(
                        f"Epoch: {epoch:4d} | "
                        f"Train: {train_acc:5.2f}% | "
                        f"Test: {test_acc:5.2f}% | "
                        f"Time: {nn._format_time(elapsed_time)} | "
                        f"Press Ctrl+C to stop"
                    )
                
                # Auto-scale axes
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                
                # Update x-axis limits if needed
                if epoch > 10:
                    ax1.set_xlim(0, epoch + 1)
                    ax2.set_xlim(0, epoch + 1)
                
                # Update plot less frequently to improve responsiveness
                if epoch % 5 == 0:  # Only update plot every 5 epochs
                    try:
                        # Make sure we have data to plot
                        if not epochs_plot or not train_acc_plot or not test_acc_plot or not loss_plot:
                            plt.pause(0.1)
                            continue
                            
                        # Make sure we're updating the right figure
                        plt.figure(fig.number)
                        
                        # Update data
                        line1.set_data(epochs_plot, train_acc_plot)
                        line2.set_data(epochs_plot, test_acc_plot)
                        line3.set_data(epochs_plot, loss_plot)
                        
                        # Adjust axis limits
                        if epochs_plot:
                            # Set x-axis limits
                            x_min, x_max = 0, max(epochs_plot) + 1
                            ax1.set_xlim(x_min, x_max)
                            ax2.set_xlim(x_min, x_max)
                            
                            # Auto-scale y-axis for accuracy (0-100%)
                            ax1.set_ylim(0, 100)
                            
                            # Auto-scale y-axis for loss (start from 0)
                            if loss_plot:
                                loss_max = max(loss_plot) * 1.1 if loss_plot else 1.0
                                ax2.set_ylim(0, max(0.1, loss_max))
                        
                        # Redraw the plot
                        ax1.relim()
                        ax1.autoscale_view()
                        ax2.relim()
                        ax2.autoscale_view()
                        
                        # Draw the plot
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        plt.pause(0.01)  # Small pause to allow the plot to update
                        
                    except Exception as e:
                        nn._log_training(f"Warning: Plot update error: {str(e)}")
                        # Reset the figure on error
                        plt.close('all')
                        plt.ion()
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                        plt.tight_layout()
                else:
                    # Small pause to prevent 100% CPU usage
                    plt.pause(0.01)
                
                # Save visualization periodically
                current_time = time.time()
                if current_time - nn.last_plot_time >= nn.plot_interval:
                    plot_path = os.path.join(nn.viz_dir, f'training_epoch_{epoch:04d}.png')
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    nn.last_plot_time = current_time
            
            # Save model and visualizations periodically
            current_time = time.time()
            save_interval_elapsed = (current_time - nn.last_save_time >= 60)
            epoch_interval_elapsed = (epoch - last_save_epoch >= nn.save_interval)
            
            if save_interval_elapsed or epoch_interval_elapsed:
                # Save model checkpoint
                model_path = nn.save_model()
                
                # Save training history plot
                if hasattr(nn, 'viz_dir'):
                    plot_path = os.path.join(nn.viz_dir, f'training_history_epoch_{epoch:04d}.png')
                    # Save the current interactive plot
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
                    # Also save the detailed history using the method
                    nn.plot_training_history(show=False, save_path=plot_path, interactive=False)
                    
                    # Save sample predictions every 5 epochs
                    if epoch % 5 == 0:
                        sample_path = os.path.join(nn.viz_dir, f'samples_epoch_{epoch:04d}.png')
                        nn.save_sample_images(X_test_flat, y_test, epoch=epoch)
                
                last_save_epoch = epoch
                nn.last_save_time = current_time
                nn._log_training(f"Checkpoint saved to: {os.path.basename(model_path)}")
            
                # Process GUI events and prevent 100% CPU usage
                plt.pause(0.1)  # This processes GUI events
                
    except KeyboardInterrupt:
        # User pressed Ctrl+C - normal termination
        nn._log_training("\n" + "="*70)
        nn._log_training("Training stopped by user (Ctrl+C)")
    except Exception as e:
        # Log any other errors
        error_msg = str(e)
        nn._log_training("\n" + "!"*70)
        nn._log_training(f"ERROR: Training failed!")
        nn._log_training("!"*70)
        nn._log_training(f"Error: {error_msg}")
        nn._log_training("\nStack trace:")
        import traceback
        nn._log_training(traceback.format_exc())
    finally:
        # Ensure we always run the signal handler to save the model
        signal_handler(None, None)
        # Restore original signal handler to allow force quit
        signal.signal(signal.SIGINT, original_sigint)

if __name__ == "__main__":
    train_continuously()
