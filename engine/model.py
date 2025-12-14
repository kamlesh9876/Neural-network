import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import datetime
from typing import List, Dict, Optional

from engine.config import TrainingConfig
from layers.dense import Dense
from layers.activation import get_activation
from layers.dropout import Dropout
from layers.batch_norm import BatchNorm
from optimizers import get_optimizer


class Model:
    """Modular neural network model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Build layers
        self.layers = []
        self.batch_norm_layers = []
        self.dropout_layers = []
        self.activation_fn = get_activation(config.activation)
        
        # Build network architecture
        layer_sizes = [config.input_size] + config.hidden_sizes + [config.output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Dense layer
            dense = Dense(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(dense)
            
            # Batch norm (not for output layer)
            if i < len(layer_sizes) - 2 and config.use_batch_norm:
                batch_norm = BatchNorm(layer_sizes[i + 1])
                self.batch_norm_layers.append(batch_norm)
            else:
                self.batch_norm_layers.append(None)
            
            # Dropout (not for output layer)
            if i < len(layer_sizes) - 2 and config.dropout_rate > 0:
                dropout = Dropout(config.dropout_rate)
                self.dropout_layers.append(dropout)
            else:
                self.dropout_layers.append(None)
        
        # Initialize optimizer
        self.optimizer = get_optimizer(
            config.optimizer, 
            config.learning_rate,
            momentum=config.momentum
        )
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.epoch_count = 0
        
        # Checkpointing
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
        self.training_log = os.path.join(self.checkpoint_dir, 'training_log.txt')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Visualization
        self.viz_dir = os.path.join(self.checkpoint_dir, 'training_visualization')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Training metadata
        self.training_start_time = None
        self.best_val_accuracy = 0.0
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the network"""
        A = X
        self.cache = {'A': [X]}
        
        for i in range(len(self.layers)):
            # Dense layer
            Z = self.layers[i].forward(A)
            self.cache[f'Z_{i}'] = Z
            
            # Activation (not for output layer)
            if i < len(self.layers) - 1:
                A = self.activation_fn.forward(Z)
                self.cache[f'A_{i+1}'] = A
                
                # Batch norm (not for output layer)
                if self.batch_norm_layers[i] is not None:
                    A = self.batch_norm_layers[i].forward(A, training=training)
                    self.cache[f'bn_{i}'] = self.batch_norm_layers[i].cache
                
                # Dropout
                if self.dropout_layers[i] is not None:
                    A = self.dropout_layers[i].forward(A, training=training)
                    self.cache[f'dropout_{i}'] = self.dropout_layers[i].mask
            else:
                # Output layer - apply softmax
                exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
                A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                self.cache['output'] = A
        
        return A
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """Backward pass and parameter update"""
        n_samples = X.shape[0]
        
        # Calculate gradient of loss with respect to output
        dZ = output - y  # Derivative of cross-entropy loss with softmax
        
        # Initialize gradients
        weights = [layer.weights for layer in self.layers]
        biases = [layer.biases for layer in self.layers]
        dW = [np.zeros_like(w) for w in weights]
        db = [np.zeros_like(b) for b in biases]
        
        # Backpropagate through the network
        for l in reversed(range(len(self.layers))):
            # Get activations from cache
            A_prev = self.cache[f'A_{l}'] if l > 0 else X
            
            # Compute gradients for weights and biases
            dW[l] = np.dot(A_prev.T, dZ) / n_samples
            db[l] = np.sum(dZ, axis=0, keepdims=True) / n_samples
            
            # Add L2 regularization
            if self.config.weight_decay > 0:
                dW[l] += self.config.weight_decay * weights[l] / n_samples
            
            # If not the first layer, compute gradient for previous layer
            if l > 0:
                # Get activation derivative
                Z_prev = self.cache[f'Z_{l-1}']
                dA = self.activation_fn.backward(Z_prev)
                
                # Apply dropout gradient (reverse order)
                if self.dropout_layers[l-1] is not None and f'dropout_{l-1}' in self.cache:
                    dA *= self.cache[f'dropout_{l-1}']
                
                # Apply batch norm gradient (reverse order)
                if self.batch_norm_layers[l-1] is not None and f'bn_{l-1}' in self.cache:
                    dZ_bn, dgamma, dbeta = self.batch_norm_layers[l-1].backward(dZ)
                    dZ = dZ_bn  # Use the gradient from batch norm
                    # Update batch norm parameters
                    self.batch_norm_layers[l-1].gamma -= self.config.learning_rate * dgamma
                    self.batch_norm_layers[l-1].beta -= self.config.learning_rate * dbeta
                else:
                    # No batch norm, dZ stays as is
                    pass
                
                # Compute gradient for previous layer
                dZ = np.dot(dZ, weights[l].T) * dA
        
        # Update weights using optimizer
        self.optimizer.update(weights, biases, dW, db)
        
        # Update layer weights
        for i, layer in enumerate(self.layers):
            layer.weights = weights[i]
            layer.biases = biases[i]
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
        """Compute cross-entropy loss with L2 regularization"""
        # Clip predictions
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Cross-entropy loss
        n_samples = y_true.shape[0]
        cross_entropy = -np.sum(y_true * np.log(y_pred)) / n_samples
        
        # L2 regularization
        l2_reg = 0
        if self.config.weight_decay > 0:
            for layer in self.layers:
                l2_reg += np.sum(layer.weights ** 2)
            l2_reg = (self.config.weight_decay / 2) * l2_reg / n_samples
        
        return cross_entropy + l2_reg
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy"""
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        return np.mean(y_pred == y_true) * 100.0
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, 
              batch_size: int = 32, verbose: bool = True) -> None:
        """Train the model for one epoch"""
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        if n_samples % batch_size != 0:
            n_batches += 1
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Forward pass
            output = self.forward(X_batch, training=True)
            
            # Compute loss and accuracy
            loss = self.compute_loss(y_batch, output)
            accuracy = self.compute_accuracy(y_batch, output)
            
            # Backward pass
            self.backward(X_batch, y_batch, output)
            
            epoch_loss += loss * (end_idx - start_idx)
            epoch_accuracy += accuracy * (end_idx - start_idx)
        
        # Update epoch metrics
        avg_loss = epoch_loss / n_samples
        avg_accuracy = epoch_accuracy / n_samples
        
        self.loss_history.append(avg_loss)
        self.accuracy_history.append(avg_accuracy)
        self.epoch_count += 1
        
        if verbose:
            print(f"Epoch {self.epoch_count}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.2f}%")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(X, training=False)
    
    def _log_training(self, message: str) -> None:
        """Log training progress"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        with open(self.training_log, 'a') as f:
            f.write(log_message + '\n')
