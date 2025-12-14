import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import argparse
import signal
import sys
from typing import Optional

from engine.config import TrainingConfig
from engine.model import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST')
    
    # Model architecture
    parser.add_argument('--input-size', type=int, default=784, help='Input size (default: 784 for MNIST)')
    parser.add_argument('--hidden-sizes', type=str, default='128,64', 
                       help='Comma-separated list of hidden layer sizes (default: 128,64)')
    parser.add_argument('--output-size', type=int, default=10, help='Output size (default: 10 for MNIST)')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD (default: 0.9)')
    parser.add_argument('--dropout-rate', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    
    # Features
    parser.add_argument('--no-batch-norm', action='store_true', help='Disable batch normalization')
    parser.add_argument('--activation', type=str, default='relu', 
                       choices=['relu', 'leaky_relu', 'elu', 'gelu'], help='Activation function (default: relu)')
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['sgd', 'adam', 'rmsprop'], help='Optimizer (default: adam)')
    
    return parser.parse_args()


def handle_signal(sig, frame):
    """Handle interrupt signals"""
    print("\nTraining interrupted. Saving checkpoint...")
    sys.exit(0)


def train_continuously():
    """Train the neural network continuously with checkpointing"""
    # Parse arguments
    args = parse_args()
    
    # Create config
    config = TrainingConfig(
        input_size=args.input_size,
        hidden_sizes=list(map(int, args.hidden_sizes.split(','))) if args.hidden_sizes else [128, 64],
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
    
    # Create model
    print("Creating model...")
    model = Model(config)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Training loop
    print("Starting continuous training...")
    model._log_training("\n" + "="*70)
    model._log_training("Starting Neural Network Training")
    model._log_training("="*70)
    model._log_training(f"Training samples: {len(X_train_flat):,}")
    model._log_training(f"Test samples: {len(X_test_flat):,}")
    model._log_training(f"Network architecture: {config.input_size}-{'-'.join(str(s) for s in config.hidden_sizes)}-{config.output_size}")
    model._log_training(f"Learning rate: {config.learning_rate}, Optimizer: {config.optimizer}")
    model._log_training(f"Batch size: {config.batch_size}, Dropout: {config.dropout_rate}")
    model._log_training(f"Batch norm: {'Enabled' if config.use_batch_norm else 'Disabled'}")
    
    model.training_start_time = time.time()
    
    # Continuous training loop
    epoch = 0
    while epoch < config.epochs:
        epoch_start_time = time.time()
        
        # Train for one epoch
        model.train(X_train_flat, y_train_cat, epochs=1, batch_size=config.batch_size, verbose=False)
        
        # Update epoch count
        epoch += 1
        
        # Evaluate on test set (sample for speed)
        if epoch % 5 == 0:  # Evaluate every 5 epochs
            sample_size = 1000
            sample_indices = np.random.choice(len(X_test_flat), sample_size, replace=False)
            
            # Calculate training accuracy
            train_output = model.predict(X_train_flat[sample_indices])
            train_acc = model.compute_accuracy(y_train_cat[sample_indices], train_output)
            
            test_output = model.predict(X_test_flat[sample_indices])
            test_acc = model.compute_accuracy(y_test_cat[sample_indices], test_output)
            
            # Calculate validation loss
            val_loss = model.compute_loss(y_test_cat[sample_indices], test_output)
            model.val_loss_history.append(val_loss)
            model.val_accuracy_history.append(test_acc)
            
            # Log progress
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - model.training_start_time
            
            model._log_training(f"Epoch {epoch:4d} | Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | Loss: {model.loss_history[-1]:.4f} | Time: {epoch_time:.2f}s")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(model.checkpoint_dir, f'model_epoch_{epoch:04d}.npz')
                
                # Save model data
                model_data = {}
                for i, layer in enumerate(model.layers):
                    model_data[f'weight_{i}'] = layer.weights
                    model_data[f'bias_{i}'] = layer.biases
                
                # Add training history
                model_data['loss_history'] = np.array(model.loss_history, dtype=np.float32)
                model_data['accuracy_history'] = np.array(model.accuracy_history, dtype=np.float32)
                model_data['val_loss_history'] = np.array(model.val_loss_history, dtype=np.float32)
                model_data['val_accuracy_history'] = np.array(model.val_accuracy_history, dtype=np.float32)
                model_data['epoch'] = np.array([epoch], dtype=np.int32)
                
                # Save batch norm parameters
                for i, bn_layer in enumerate(model.batch_norm_layers):
                    if bn_layer is not None:
                        model_data[f'bn_gamma_{i}'] = bn_layer.gamma
                        model_data[f'bn_beta_{i}'] = bn_layer.beta
                        model_data[f'running_mean_{i}'] = bn_layer.running_mean
                        model_data[f'running_var_{i}'] = bn_layer.running_var
                
                np.savez_compressed(checkpoint_path, **model_data)
                
                # Save training history as JSON
                history = {
                    'epoch': int(epoch),
                    'loss': [float(loss) for loss in model.loss_history],
                    'accuracy': [float(acc) for acc in model.accuracy_history],
                    'val_loss': [float(loss) for loss in model.val_loss_history],
                    'val_accuracy': [float(acc) for acc in model.val_accuracy_history],
                    'timestamp': float(time.time()),
                    'training_time': float(time.time() - model.training_start_time),
                    'config': {
                        'learning_rate': float(config.learning_rate),
                        'batch_size': int(config.batch_size),
                        'epochs': int(config.epochs),
                        'optimizer': str(config.optimizer),
                        'use_batch_norm': bool(config.use_batch_norm),
                        'dropout_rate': float(config.dropout_rate)
                    }
                }
                
                history_path = os.path.join(model.checkpoint_dir, 'training_history.json')
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)
                
                model._log_training(f"Checkpoint saved to {checkpoint_path}")
        
        # Plot training history every 30 epochs
        if epoch % 30 == 0 and epoch > 0:
            try:
                plot_path = os.path.join(model.viz_dir, f'training_history_{epoch}.png')
                
                plt.figure(figsize=(15, 5))
                
                # Plot Loss
                plt.subplot(1, 3, 1)
                plt.plot(model.loss_history, 'b-', label='Training Loss')
                if model.val_loss_history:
                    plt.plot(model.val_loss_history, 'r-', label='Validation Loss')
                plt.title('Training & Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot Accuracy
                plt.subplot(1, 3, 2)
                plt.plot(model.accuracy_history, 'g-', label='Training Accuracy')
                if model.val_accuracy_history:
                    plt.plot(model.val_accuracy_history, 'r-', label='Validation Accuracy')
                plt.title('Training & Validation Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add training info
                plt.subplot(1, 3, 3)
                plt.axis('off')
                info_text = f"""
                Training Progress
                
                Epoch: {epoch}
                Learning Rate: {config.learning_rate}
                Optimizer: {config.optimizer}
                Batch Size: {config.batch_size}
                
                Current Accuracy:
                Train: {model.accuracy_history[-1]:.2f}%
                Test: {model.val_accuracy_history[-1] if model.val_accuracy_history else 0:.2f}%
                
                Training Time: {time.time() - model.training_start_time:.1f}s
                """
                plt.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                model._log_training(f"Training plot saved to {plot_path}")
                
            except Exception as e:
                model._log_training(f"Error plotting training history: {e}")


if __name__ == "__main__":
    train_continuously()
