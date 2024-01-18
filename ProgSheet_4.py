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

# # 

# # Programming Assignment 4
#
# Written by
# Ramon Leiser
# Tor Groje

# # Provided Functions

# +
import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spla

from scipy.spatial import Delaunay

import matplotlib.pyplot as plt


from Ex4_provided_functions import *

# print(boundary_edges)

# ## (a)

def assemble_neumann_vector_local(vertex_coords):
    return np.linalg.norm(vertex_coords[0] - vertex_coords[1]) / 2

def assemble_neumann_vector(neumann_edges, nodes, neumann_fn):
    neumann_vector = np.zeros(len(nodes))
    for edge in neumann_edges:
        edge_vertex_coords = nodes[edge]
        x_edge = (edge_vertex_coords[0] + edge_vertex_coords[1]) / 2
        edge_value = neumann_fn(x_edge[0], x_edge[1]) * assemble_neumann_vector_local(edge_vertex_coords)
        neumann_vector[edge] += edge_value
    return neumann_vector
 
 # ## (b)
 
def solve(A, b, dirichlet_nodes, nodes, dirichlet_fn):
    
    Iu_D = np.zeros(len(nodes))
    Iu_D[dirichlet_nodes] = dirichlet_fn( nodes[dirichlet_nodes][:,0], nodes[dirichlet_nodes][:,1] ) 

    interior = list(set(np.arange(len(nodes))) - set(dirichlet_nodes));
    b_inner = b[interior]
    A_inner = A[tuple(np.meshgrid(interior, interior, indexing='ij'))]

    b_til = b_inner - A.dot(Iu_D)[interior]

    u = np.zeros(len(nodes));
    
    # print(sp.linalg.spsolve(A_inner, b_til.astype(np.float64)))

    u[dirichlet_nodes] = Iu_D[dirichlet_nodes]
    u[interior] = sp.linalg.spsolve(A_inner, b_til.astype(np.float64));
    return u;

### Ex 3
from ProgSheet_3 import assemble_mass_matrix, assemble_stiffness_matrix, ref_map

def f(x,y):
    return np.pi**2 / 4 * np.cos(np.pi * y / 2) - 1;

def g(x,y):
    return 1 / np.sqrt(5.) * (3. / 5 + 2 * np.abs(x) - np.pi / 2 * np.sin(np.pi * y / 2));

def u_D(x, y):
    return x**2 / 2. + 3. * y / 5 + np.cos(np.pi * y / 2);

# ## (c)
def assemble_christmas_tree_problem(nodes, elements, boundary_nodes, boundary_edges):
    a = nodes[boundary_edges][:,0]
    b = nodes[boundary_edges][:,1]
    
    # Neumann edges are those where the y coordinate changes
    x = (a[:,1] - b[:,1] != 0)
    neumann_edges = np.array(boundary_edges)[x]
    
    # Direchlet nodes are those with y coordinate in 0, 2, 4 and 6
    y_coords = nodes[boundary_nodes][:,1] 
    x = (y_coords == 0);
    for k in [2,4,6]:
        x += (y_coords == k)

    dirichlet_nodes = np.array(boundary_nodes)[x]
    
    A = assemble_stiffness_matrix(elements, nodes)
    M = assemble_mass_matrix(elements, nodes)

    b_base = M.dot( f(nodes[:,0], nodes[:,1]) )
    b_neumann = assemble_neumann_vector(neumann_edges, nodes, g)
    
    # print( sum(A.dot(np.ones(len(nodes)))) )
    return A, M, neumann_edges, dirichlet_nodes, b_base + b_neumann

def plot_solution(nodes, elements, u):
    x = nodes[:,0]
    y = nodes[:,1]

    from matplotlib.tri import Triangulation
    triangulation = Triangulation(x, y, elements)

    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tpc = ax1.tripcolor(triangulation, u) #, shading='flat', vmin=0, vmax=5.5)
    fig1.colorbar(tpc)
    plt.show()

# if __name__ == "__main__":
#     nodes, elements, boundary_nodes, boundary_edges = mesh = christmas_tree_mesh(k = 2)
#     A, M, neumann_edges, dirichlet_nodes, b \
#             = assemble_christmas_tree_problem(*mesh)
#     u = solve(A, b, dirichlet_nodes, nodes, u_D)
#     plot_solution(nodes, elements, u)

# ## (d)

def grad_u_fn(x,y):
    # grad_u = np.zeros((len(x),2))
    return np.array([x, 3./5 - np.pi/2*np.sin(np.pi * y / 2)])

def H1_error(elements, nodes, u, grad_u_fn):
    error = 0.0;
    preG = np.zeros((3,3))
    preG[0,:] = 1;
    for elem in elements:
        vertex_coords = nodes[elem]
        B, trans = ref_map(vertex_coords)
        preG[1,:] = vertex_coords[:,0]
        preG[2,:] = vertex_coords[:,1]
        
        preGinv = np.linalg.inv(preG)
        G = preGinv.dot( np.array([[0,0], [1,0], [0,1]]) );

        x_elem = np.sum(vertex_coords, axis=0) / 3.

        grad_u_T = grad_u_fn(x_elem[0], x_elem[1])
        grad_uh_T = u[elem].dot(G)


        error += .5 * np.linalg.det(B) \
                 * np.linalg.norm(grad_u_T - grad_uh_T)**2
        # print( .5 * np.linalg.det(B) \
        #         * np.linalg.norm(grad_u_T - grad_uh_T)**2 )

    return np.sqrt( error )

# ## From Ex3 

def L2_error(M, nodes, u, u_fn):
    diff = u_fn(nodes[:,0], nodes[:,1]) - u
    return np.sqrt( diff.dot(M.dot(diff)) )
    # return np.sqrt( diff.dot(M.dot(diff)) )


# ## (e)

if __name__ == "__main__":
    l2_err = []
    h1_err = []

    for k in [2, 3, 4, 5]:
        nodes, elements, boundary_nodes, boundary_edges = mesh = christmas_tree_mesh(k)
        A, M, neumann_edges, dirichlet_nodes, b \
                = assemble_christmas_tree_problem(*mesh)
        u = solve(A, b, dirichlet_nodes, nodes, u_D)

        # plot_solution(nodes, elements, u)
        # plot_solution(nodes, elements, u_D(nodes[:,0], nodes[:,1]))
        h1_err.append( H1_error(elements, nodes, u, grad_u_fn) )
        l2_err.append( L2_error(M, nodes, u, u_D) )

        print("k = %d; done" % k)
    print("L2: ", l2_err)
    print("H1: ", h1_err)

    plt.plot(np.arange(len(l2_err)), np.array(l2_err))
    plt.plot(np.arange(len(h1_err)), np.array(h1_err))


    plt.show()



