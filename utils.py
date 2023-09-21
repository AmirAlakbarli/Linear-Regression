import numpy as np
import matplotlib.pyplot as plt
import math
import copy


def generate_w_b(*w_shape, w_limit=1, b_limit=1):
    w = 2 * w_limit * np.random.rand(*w_shape) - w_limit
    b = 2 * b_limit * np.random.rand() - b_limit
    return w, b


def generate_X_y(shape, w, b, x_limit=1, random_err=0):
    X = 2 * x_limit * np.random.rand(*shape) - x_limit
    m = X.shape[0]
    random_err = 2 * random_err * np.random.rand(m) - random_err
    y = np.dot(X, w) + b + random_err
    return X, y


def compute_cost(X, y, w, b):
    m = X.shape[0]
    return np.sum((np.dot(X, w) + b - y) ** 2) / (2 * m)


def compute_gradient(X, y, w, b):
    m = X.shape[0]
    dj_dw = np.dot(np.dot(X, w) + b - y, X) / m
    dj_db = np.sum(np.dot(X, w) + b - y) / m
    return dj_dw, dj_db


def mse(y, y_hat):
    m = y.shape[0]
    return np.sum((y_hat - y) ** 2) / m


def mae(y, y_hat):
    m = y.shape[0]
    return np.sum(np.abs(y_hat - y)) / m


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i < 100000:
            J_history.append(cost_function(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i: 4d}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history


def plt_linear(x, y):
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y)
    # give name to the axis
    plt.xlabel('x')
    plt.ylabel('y')
    # divide axis value by 10 and show 1e2 instead of 100
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
