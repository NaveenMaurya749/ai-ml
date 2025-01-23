import numpy as np
import matplotlib.pyplot as plt

# Gradient Descent for Optimization of C^1 functions from R^n to R

class vector:
    coords = []
    dim = 0

    def __init__(self, coords):
        self.coords = coords
        self.dim = len(coords)

    def __add__(self, other):
        result = []
        for i in range(self.dim):
            result.append(self.coords[i] + other.coords[i])
        return vector(result)
    
    def __mul__(self, scalar):
        result = []
        for i in range(self.dim):
            result.append(scalar * self.coords[i])
        return vector(result)
    
    def __sub__(self, other):
        return self + other * (-1)
    
    def append(self, value):
        self.coords.append(value)
        self.dim += 1

def random_vector(dim):
    return vector(np.random.rand(dim))

def func(input : vector) -> float:
    result = 0
    denom = 0
    for i in range(input.dim):
        result *= input.coords[i]
        denom += input.coords[i]**2
    if denom == 0:
        return 0
    result /= denom
    return result

def grad(fun, input : vector):
    epsilon = 1e-5
    result = vector([0] * input.dim)
    for i in range(input.dim):
        # Partial derviates calculated for the derivative by symmetric difference quotient
        change = vector([0] * input.dim)
        change.coords[i] = epsilon
        result.coords[i] = (fun(input + change) - fun(input - change)) / (2 * epsilon)
    return result

def grad_descent(fun, input : vector, alpha = 0.01, max_iter = 1000, tol = 1e-5):
    for i in range(max_iter):
        gradient = grad(fun, input)
        if np.linalg.norm(gradient.coords) < tol:
            break
        input = input - gradient * alpha
    return input

# # Test the gradient descent
# print(grad_descent(func, random_vector(10)).coords)
for i in range(10):
    print(grad_descent(func, random_vector(10)).coords)