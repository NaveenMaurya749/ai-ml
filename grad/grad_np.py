import numpy as np

def random_vector(dim):
    return np.random.rand(dim)

# The desired float-valued C^1 function to be minimized
def func(input):
    # result = np.prod(input)
    # denom = np.sum(input**2)
    # if denom == 0:
    #     return 0
    # return result / denom
    return np.sum((input-[2.0]*len(input))**2)

# Numerical computation of gradient of a function at a point
def grad(fun, input):
    epsilon = 1e-5
    result = np.zeros_like(input)
    for i in range(len(input)):
        change = np.zeros_like(input)
        change[i] = epsilon
        result[i] = (fun(input + change) - fun(input - change)) / (2 * epsilon)
    return result

def grad_descent(fun, input, alpha=0.01, max_iter=1000, tol=1e-5):
    for i in range(max_iter):
        gradient = grad(fun, input)
        if np.linalg.norm(gradient) < tol:
            break
        input = input - alpha * gradient
    return input

def square_minus_two(z):
    return [z[0]**2 - z[1]**2, 2*z[0]*z[1]]

# Test the gradient descent
for i in range(10):
    print(grad_descent(func, random_vector(100)))