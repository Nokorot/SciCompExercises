# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, kron, eye, linalg


def A_matrix(n):
    ones = np.ones(n-1);
    L = np.sqrt(n)*spdiags([-ones, 2*ones, ones], [-1,0,1])
    I = eye(n-1)
    return kron(I,L) + kron(L,I)


def jacobi_iteration(A, u0, b, eps):
    Dinv = np.diag(1/A.diagonal())
    alpha = Dinv.dot(b - A.dot(u0))
    uk = u0 + alpha
    while np.sqrt(alpha.dot(alpha)) > eps:
        alpha = Dinv.dot(b - A.dot(uk))
        uk = uk + alpha
    return uk


def ploblem_1_solve(n):
    print(f"Debug: Solving problem 1 for n={n}")
    A = A_matrix(n)
    u0 = np.zeros((n-1)*(n-1))

    x = y = np.linspace(0,1,n+1)[1:-1]
    X, Y = np.meshgrid(x,y)

    f = 5 * np.pi**2 * np.sin(2*np.pi*X)*np.sin(np.pi*Y)
    b = f.flatten()

    return linalg.spsolve(A,b), (X,Y)
    u = jacobi_iteration(A, u0, b, 1e-10)
    return u, (X,Y)


# + editable=true slideshow={"slide_type": ""}
def problem_1_plot(u, XY):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    ax.plot_surface(XY[0],XY[1],u.reshape(XY[0].shape))
    return fig


# -

def problem_1_absolut_error(n, u_approx, XY):
    X, Y = XY
    u = np.sin(2*np.pi*X)*np.sin(np.pi*Y)
    return np.max(np.abs(u_approx - u.flatten()))


ns = [8,16,32,64,128]
us = [ ploblem_1_solve(n) for n in ns ]

fig = problem_1_plot(*us[-1])


u_ = ploblem_1_solve(n)
fig = problem_1_plot(u_)

errors = [ problem_1_absolut_error(n, *un) for n, u in zip(ns,us) ]
