import numpy as np
from engine.autograd.tensor import Tensor, tensor

def test_scalar_operations():
    print('Testing scalar operations with autograd...')
    x = tensor(2.0)
    y = tensor(3.0)
    z = x + y
    print(f'x = {x.value}, y = {y.value}, z = {z.value}')
    z.backward()
    print(f'dz/dx = {x.grad}, dz/dy = {y.grad}')

def test_multiplication():
    print('Testing multiplication...')
    x = tensor(4.0)
    y = tensor(5.0)
    z = x * y
    print(f'x = {x.value}, y = {y.value}, z = {z.value}')
    z.backward()
    print(f'dz/dx = {x.grad}, dz/dy = {y.grad}')

def test_matrix_multiplication():
    print('Testing matrix multiplication...')
    X = tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    Y = tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
    Z = X @ Y
    print(f'X = {X.value}')
    print(f'Y = {Y.value}')
    print(f'Z = X @ Y = {Z.value}')
    loss = Z.sum()
    loss.backward()
    print(f'dL/dX = {X.grad}')
    print(f'dL/dY = {Y.grad}')

if __name__ == '__main__':
    test_scalar_operations()
    test_multiplication()
    test_matrix_multiplication()
