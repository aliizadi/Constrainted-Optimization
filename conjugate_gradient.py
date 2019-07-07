import numpy as np
from numpy.linalg import norm
import time


ITERATION = 100
EPSILON = .00000001

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


def linear_conjugate_gradient(A, x0, b):
    x = x0
    r = np.matmul(A, x) - b
    p = -r
    k = 0

    start_time = time.time()

    while not (k > ITERATION or norm(r) < EPSILON):
        pAp = np.matmul(np.matmul(p.T, A), p)
        alpha = - np.matmul(r.T, p) / pAp
        x = x + alpha * p
        r = np.matmul(A, x) - b
        beta = np.matmul(np.matmul(r.T, A), p) / pAp
        p = -r + beta * p
        k = k + 1

    print("--- Algorithm runtime: %s seconds ---" % (time.time() - start_time))
    return x


n = [100, 400, 1600]
A_hilb = list(map(lambda j: hilb(j), n))
A_tridiag = list(map(lambda j: tridiag(j), n))
b_init = list(map(lambda j: bias(j), n))
x_init = list(map(lambda j: np.random.rand(j, 1), n))

x_stars_hlib = []
x_stars_tridiag = []

for i in range(len(n)):
    x_star = linear_conjugate_gradient(A_tridiag[i], x_init[i], b_init[i])
    x_stars_tridiag.append(x_star)
    print("Residual of tridiag matrix with size: %s is %s " % (n[i], norm(np.matmul(A_tridiag[i], x_star) - b_init[i])), "\n")

for i in range(len(n)):
    x_star = linear_conjugate_gradient(A_hilb[i], x_init[i], b_init[i])
    x_stars_hlib.append(x_star)
    print("Residual of hilb matrix with size: %s is %s " % (n[i], norm(np.matmul(A_hilb[i], x_star) - b_init[i])), "\n")


epsilon = 1e-6

epsilon_init = list(map(lambda j: np.random.rand(j, 1) * epsilon, n))

A_hat_hilb = list(map(lambda j: A_hilb[j] + epsilon_init[j], [i for i in range(len(n))]))
A_hat_tridiag = list(map(lambda j: A_tridiag[j] + epsilon_init[j], [i for i in range(len(n))]))


print("---------- stochastic part ----------", "\n")

for i in range(len(n)):
    x_star = linear_conjugate_gradient(A_hat_tridiag[i], x_init[i], b_init[i])
    print("Residual of stochastic tridiag matrix with size: %s is %s " % (n[i], norm(np.matmul(A_hat_tridiag[i], x_star) - b_init[i])), "\n")
    print("delta A: ", norm(A_tridiag[i] - A_hat_tridiag[i]))
    print("delta x star: ", norm(x_stars_tridiag[i] - x_star))

for i in range(len(n)):
    x_star = linear_conjugate_gradient(A_hat_hilb[i], x_init[i], b_init[i])
    print("Residual of stochastic hilb matrix with size: %s is %s " % (n[i], norm(np.matmul(A_hat_hilb[i], x_star) - b_init[i])), "\n")
    print("delta A: ", norm(A_hilb[i] - A_hat_hilb[i]))
    print("delta x star: ", norm(x_stars_hlib[i] - x_star))











