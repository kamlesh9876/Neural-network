import numpy as np
from engine.config import TrainingConfig
from engine.model import Model

def test_modular_structure():
    """Test modular structure without TensorFlow dependencies"""
    print("Testing modular neural network structure...")
    
    # Create simple config
    config = TrainingConfig(
        input_size=784,
        hidden_sizes=[64, 32],
        output_size=10,
        learning_rate=0.001,
        batch_size=32,
        epochs=2,
        dropout_rate=0.2,
        use_batch_norm=False,  # Disable for now
        activation='relu',
        optimizer='adam'
    )
    
    # Create model
    print("Creating model...")
    model = Model(config)
    
    # Create dummy data
    print("Creating dummy data...")
    X_dummy = np.random.randn(100, 784)
    y_dummy = np.random.randn(100, 10)
    
    # Test forward pass
    print("Testing forward pass...")
    output = model.forward(X_dummy, training=True)
    print(f"Forward pass output shape: {output.shape}")
    
    # Test backward pass
    print("Testing backward pass...")
    model.backward(X_dummy, y_dummy, output)
    print("Backward pass completed!")
    
    # Test training
    print("Testing training...")
    model.train(X_dummy, y_dummy, epochs=1, batch_size=16, verbose=True)
    print("Training completed!")
    
    print("✅ Modular structure test passed!")

if __name__ == "__main__":
    test_modular_structure()
