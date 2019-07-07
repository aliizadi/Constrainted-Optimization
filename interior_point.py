import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

NORM_EPSILON = 0.001
ETTA_EPSILON = 0.1

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


def quadratic_grad(A, x, b):
    return np.matmul(A, x) - b


def constraint_grad(P):
    return P


def quadratic_hessian(A):
    return A


def r_dual(A, x, b, P, q, lam):
    return quadratic_grad(A, x, b) + np.matmul(P.T, lam)


def r_cent(x, P, q, lam, t):
    return - np.matmul(np.diag(np.reshape(lam, -1)), (np.matmul(P, x) - q)) - np.ones([np.shape(P)[0], 1]) / t


def residual(A, x, b, P, q, lam, t):
    return norm(np.append(r_dual(A, x, b, P, q, lam), r_cent(x, P, q, lam, t), axis=0))


def sufficient_decrease_condition(A, x, b, alpha, c, P, q, lam, delta_x, delta_lam, t):
    next_step_x = x + alpha * delta_x
    next_step_lam = lam + alpha * delta_lam
    return np.min(next_step_lam) > 0 > np.max(np.matmul(P, next_step_x) - q) and \
        residual(A, next_step_x, b, P, q, next_step_lam, t) <= (1 - c * alpha) * residual(A, x, b, P, q, lam, t)


def line_search(A, x, b, P, q, lam, delta_x, delta_lam, t):
    alpha = 1
    r = 0.5
    c = 0.5

    backtrack_iteration = 0
    while not (sufficient_decrease_condition(A, x, b, alpha, c, P, q, lam, delta_x, delta_lam, t) or alpha < 0.001):
        alpha = alpha * r
        backtrack_iteration += 1
        # print(backtrack_iteration)

    return alpha


def primal_dual_interior_point(A, x, b, P, q, lam, mu):
    duality_gaps = []
    r_feas = []
    iterations = []
    m, n = np.shape(P)
    iteration = 0

    while True:
        iteration += 1
        iterations.append(iteration)

        etta = - np.matmul((np.matmul(P, x) - q).T, lam)[0, 0]

        t = mu * m / etta
        duality_gaps.append(etta)

        r_dual_norm = norm(r_dual(A, x, b, P, q, lam))

        r_feas.append(r_dual_norm)

        factor_matrix = np.vstack([np.hstack([quadratic_hessian(A), P.T]), np.hstack([-np.matmul(np.diag(np.reshape(lam, -1)), P.T), -np.diag(np.reshape(np.matmul(P, x) - q, -1))])])
        residual_matrix = - np.vstack([r_dual(A, x, b, P, q, lam), r_cent(x, P, q, lam, t)])
        delta_matrix = np.matmul(np.linalg.pinv(factor_matrix), residual_matrix)

        delta_x = delta_matrix[:n]
        delta_lam = delta_matrix[n:]

        s = line_search(A, x, b, P, q, lam, delta_x, delta_lam, t)

        x = x + s * delta_x
        lam = lam + s * delta_lam

        print(iteration)
        print(etta)
        print(r_dual_norm)
        print('----------------')

        if r_dual_norm <= NORM_EPSILON or etta <= ETTA_EPSILON:
            return x, iterations, duality_gaps, r_feas


n_list = [100, 400]
A_hilb = list(map(lambda j: hilb(j), n_list))
A_tridiag = list(map(lambda j: tridiag(j), n_list))
b_init = list(map(lambda j: bias(j).reshape(1, -1).T, n_list))
x_init = list(map(lambda j: np.random.rand(j, 1) * 10e-3, n_list))
P_init = list(map(lambda j: np.random.rand(j, j), n_list))
q_init = list(map(lambda j: (np.random.rand(j, 1) + np.ones([j, 1])), n_list))
lam_init = list(map(lambda j: np.random.rand(j, 1) + .1, n_list))


for i in range(len(n_list)):
    x_star, iters, d_g, r_f = primal_dual_interior_point(A_tridiag[i], x_init[i], b_init[i], P_init[i], q_init[i], lam_init[i], mu=10)
    fig1 = plt.figure()
    plt.plot(iters, d_g)
    plt.xlabel('Iterations')
    plt.ylabel('Duality gap')
    plt.title('Primal Dual Interior Point')

    fig2 = plt.figure()
    plt.plot(iters, r_f)
    plt.xlabel('Iterations')
    plt.ylabel('Feasibility Residual')
    plt.title('Primal Dual Interior Point')

for i in range(len(n_list)):
    x_star, iters, d_g, r_f = primal_dual_interior_point(A_tridiag[i], x_init[i], b_init[i], P_init[i], q_init[i], lam_init[i], mu=10)
    fig1 = plt.figure()
    plt.plot(iters, d_g)
    plt.xlabel('Iterations')
    plt.ylabel('Surrogate Duality gap')
    plt.title('Primal Dual Interior Point')

    fig2 = plt.figure()
    plt.plot(iters, r_f)
    plt.xlabel('Iterations')
    plt.ylabel('Feasibility Residual')
    plt.title('Primal Dual Interior Point')

plt.show()













