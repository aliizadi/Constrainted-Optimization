import numpy as np
import matplotlib.pyplot as plt


def bias(n):
    return np.ones(n)


def tridiag(n, d1=-1, d2=4, d3=-1):
    a = np.zeros((n, n), int)
    d1s = np.repeat(d1, n-1)
    d3s = np.repeat(d3, n-1)
    np.fill_diagonal(a, d2)
    np.fill_diagonal(a[1:], d1s)
    np.fill_diagonal(a[:, 1:], d3s)
    return a


def hilb(n):
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a[i, j] = (i + j + 1)
    return a


def quadratic(A, x, b):
    return 0.5 * np.matmul(np.matmul(x.T, A), x) - np.matmul(b.T, x)


def quadratic_grad(A, x, b):
    return np.matmul(A, x) - b


def quadratic_hessian(A):
    return A


def phi(P, x, q):
    # return -np.sum(np.log(- (np.matmul(P, x) - q)))
    r = 0
    for i in range(np.shape(q)[0]):
        pi = np.reshape(P[i], (1, -1))
        qi = np.reshape(q[i], (1, 1))
        fi = np.matmul(pi, x) - qi
        if fi >= 0:
            r -= - 1 / .0000001
        else:
            r -= np.log(-fi)

    return r


def phi_grad(P, x, q):
    return np.sum(P.T / -(np.matmul(P, x) - q), axis=0).reshape((1, -1)).T


def phi_hessian(P, x, q):
    fi = np.matmul(P, x) - q
    return np.matmul(P.T, P) / (fi * fi)


def objective(A, x, b, P, q, t):
    return t * quadratic(A, x, b) + phi(P, x, q)


def objective_grad(A, x, b, P, q, t):
    return t * quadratic_grad(A, x, b) + phi_grad(P, x, q)


def objective_hessian(A, x, b, P, q, t):
    return t * quadratic_hessian(A) + phi_hessian(P, x, q)


def newton_direction(A, x, b, P, q, t):
    return - np.matmul(np.linalg.pinv(objective_hessian(A, x, b, P, q, t)), objective_grad(A, x, b, P, q, t))


def sufficient_decrease_condition(A, x, b, alpha, c, p, P, q, t):
    next_step_x = x + alpha * p
    return objective(A, next_step_x, b, P, q, t) <= objective(A, x, b, P, q, t) + c * alpha * np.dot(objective_grad(A, x, b, P, q, t).T, p)


def backtracking_line_search(A, x, b, p, P, q, t):
    alpha = 1
    r = 0.5
    c = 0.5

    backtrack_iteration = 0
    while not sufficient_decrease_condition(A, x, b, alpha, c, p, P, q, t):
        alpha = alpha * r
        backtrack_iteration += 1

    return alpha


def newton_stopping_criterion(newton_step, A, x, b, P, q, t):
    return abs(np.matmul(np.matmul(newton_step.T, objective_hessian(A, x, b, P, q, t)), newton_step))


def newton_method(A, x0, b, P, q, t):
    newton_step = newton_direction(A, x0, b, P, q, t)
    epsilon = 20
    x = x0

    iteration = 0

    while newton_stopping_criterion(newton_step, A, x, b, P, q, t)/2 > epsilon:

        step_size = backtracking_line_search(A, x, b, newton_step, P, q, t)
        x = x + step_size * newton_step
        newton_step = newton_direction(A, x, b, P, q, t)
        iteration += 1

    return x, iteration


def log_barrier(A, x, b, P, q, mu=1.5, t=1):

    newton_iterations = []
    duality_gaps = []
    m = np.shape(q)[0]
    tolerance = 0.1

    algorithm_iteration = 0
    while True:
        x, iteration = newton_method(A, x, b, P, q, t)
        newton_iterations.append(iteration)
        duality_gap = m / t
        duality_gaps.append(duality_gap)

        algorithm_iteration += 1

        if duality_gap < tolerance:
            return x, newton_iterations, duality_gaps

        t = t * mu



n = [100]
A_hilb = list(map(lambda j: hilb(j), n))
A_tridiag = list(map(lambda j: tridiag(j), n))
b_init = list(map(lambda j: bias(j).reshape(1, -1).T, n))
x_init = list(map(lambda j: np.ones([j, 1]) + np.random.rand(j, 1), n))
P_init = list(map(lambda j: np.random.rand(j, j), n))
q_init = list(map(lambda j: (np.random.rand(j, 1)), n))


for i in range(len(n)):
    x_star, ni, dg = log_barrier(A_tridiag[i], x_init[i], b_init[i], P_init[i], q_init[i])
    plt.plot(ni, dg, label=1)
    plt.xlabel('Newton Iterations')
    plt.ylabel('Duality gap')
    plt.show()









