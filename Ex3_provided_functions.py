# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Programming Exercise 3

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def criss_cross_mesh(n):
    ## Generates a criss-cross mesh of the unit square with n+1 nodes in each direction
    
    ## Returns: nodes:          an array with the coordinates of every node, i.e. nodes[k] = [x_k, y_k]
    ##          elements:       an array of the elements, each of them represented by the indices (in nodes) of its 3 vertices
    ##          boundary_nodes: a list of the indices (in nodes) of those nodes in the boundary of the domain
    
    x, y = np.meshgrid(np.linspace(0,1,n+1), np.linspace(0,1,n+1))
    # Reshape as 1D arrays
    x = x.ravel()[:, np.newaxis]
    y = y.ravel()[:, np.newaxis]
    
    nodes = np.hstack([x, y])
    
    elements = []
    for j in range(n):
        for i in range(n):
            # Add triangles from the box with lower-left corner (i/(n+1), j/(n+1))
            ###      d ------- c
            ###      |      /  |
            ###      |    /    |
            ###      |  /      |
            ###      a ------- b
            a = j*(n+1)+i
            b = a + 1
            c = b + n + 1
            d = a + n + 1
            
            elements.append([a, b, c]) # Lower-right triangle
            elements.append([a, c, d]) # Upper-left triangle
    
    elements = np.asarray(elements)
    
    boundary_nodes = []
    for k in range(n+1):
        boundary_nodes.append(k)         # Bottom side y = 0
        boundary_nodes.append(n*(n+1)+k) # Top side y = 1
    for k in range(1,n):
        boundary_nodes.append(k*(n+1))   # Left side x = 0
        boundary_nodes.append(k*(n+1)+n) # Right side x = 1
    
    return nodes, elements, boundary_nodes


# +
# Visualize the mesh for n = 3
nodes, elements, boundary_nodes = criss_cross_mesh(3)

fig = plt.figure(figsize=(6,6))

plt.triplot(nodes[:,0], nodes[:,1], elements) # Triangles
plt.plot(nodes[:,0], nodes[:,1], 'ro') # Nodes
plt.plot(nodes[boundary_nodes,0], nodes[boundary_nodes,1], 'go', mfc='none') # Boundary nodes

for j, p in enumerate(nodes):
    plt.text(p[0]-0.03, p[1]+0.03, j, color='r', ha='right') # label the points

for j, s in enumerate(elements):
    p = nodes[s].mean(axis=0)
    plt.text(p[0], p[1], '{}'.format(j), ha='center') # label triangles

plt.xlim(-0.1, 1.1); plt.ylim(-0.1, 1.1)
plt.show()


# -

# ## (a)

# +
def assemble_stiffness_matrix(elements, nodes):
    # Empty sparse matrix 
    A = sp.lil_matrix((len(nodes), len(nodes)))
    
    # Loop over all elements 
    for idx in elements:
        A[tuple(np.meshgrid(idx, idx, sparse=True))] += assemble_stiffness_matrix_local(nodes[idx])
    
    return sp.csr_matrix(A)

def assemble_mass_matrix(elements, nodes):
    M = sp.lil_matrix((len(nodes),len(nodes)))
    
    for idx in elements:
        M[tuple(np.meshgrid(idx, idx, sparse=True))] += assemble_mass_matrix_local(nodes[idx])
        
    return sp.csr_matrix(M)
