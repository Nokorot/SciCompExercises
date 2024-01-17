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

# # Programming Assignment 2
#
# Written by
# Ramon Leiser
# Tor Groje

# # %load_ext line_profiler
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as s


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
# nodes, elements, boundary_nodes = criss_cross_mesh(3)
# 
# fig = plt.figure(figsize=(6,6))
# 
# plt.triplot(nodes[:,0], nodes[:,1], elements) # Triangles
# plt.plot(nodes[:,0], nodes[:,1], 'ro') # Nodes
# plt.plot(nodes[boundary_nodes,0], nodes[boundary_nodes,1], 'go', mfc='none') # Boundary nodes
# 
# for j, p in enumerate(nodes):
#     plt.text(p[0]-0.03, p[1]+0.03, j, color='r', ha='right') # label the points
# 
# for j, s in enumerate(elements):
#     p = nodes[s].mean(axis=0)
#     plt.text(p[0], p[1], '{}'.format(j), ha='center') # label triangles
# 
# plt.xlim(-0.1, 1.1); plt.ylim(-0.1, 1.1)
# plt.show()


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




N = 10
nodes, elements, boundary_nodes = criss_cross_mesh(N)


def ref_map(vertex_coords):
    [ a, b, c ] = vertex_coords;
    F = [ [b[0] - a[0], b[1] - a[1]],
          [c[0] - a[0], c[1] - a[1]] ];
    return np.matrix(F), a;

M0 = 1/24.0 * np.matrix([[2.0, 1.0, 1.0], 
                         [1.0, 2.0, 1.0], 
                         [1.0, 1.0, 2.0]]);

def assemble_mass_matrix_local(vertex_coords):
    F, tras = ref_map(vertex_coords);
    Fdet = abs(np.linalg.det(F));
    return Fdet * M0;

grad_phi = np.matrix([[ -1, -1 ], [ 1, 0 ], [ 0, 1 ]]);
def assemble_stiffness_matrix_local(vertex_coords):
    F, tras = ref_map(vertex_coords);
    Finv = np.linalg.inv(F).T;

    interm = [ np.dot(Finv, grad_phi[i].T) for i in range(3) ];
    
    A = np.zeros((3,3))
    for i in range(3):
        A[i,i] = np.dot(interm[i].T, interm[i]);
        for j in range(i+1, 3):
            A[i,j] = A[j,i] = np.dot(interm[i].T, interm[j]);
    return A;

def solve(A, b, dirichlet_nodes):
    interior = list(set(np.arange(len(b))) - set(dirichlet_nodes));
    A = A[tuple(np.meshgrid(interior, interior, sparse=True))]
    b = b[interior];

    u = np.zeros(len(b)**2);
    u[interior] = sp.linalg.spsolve(A, b.astype(np.float64));
    return u;

    # A[tuple(np.meshgrid(dirichlet_nodes, dirichlet_nodes, sparse=True))] \
    #         = sp.eye(len(dirichlet_nodes));
    # b[dirichlet_nodes] = 0;

    # return sp.linalg.spsolve(A, b.astype(np.float64));
    # sp.linalg.spsolve()

def plot_solution(x,y, u):
    # triangulation = plt.Triangulation()j

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    cmap = plt.get_cmap('hot');
    ax.plot_trisurf(x, y, u, cmap = cmap);
    return fig
    # ax.plot_surface(XY[0],XY[1],u.reshape(XY[0].shape))

def compute_I_f(elements, nodes):
    x = nodes[:, 0];
    y = nodes[:, 1];

    return 5 * np.pi**2 * np.sin(2*np.pi*x)*np.sin(np.pi*y)


if __name__ == "__main__":
    A = assemble_stiffness_matrix(elements, nodes)
    M = assemble_mass_matrix(elements, nodes)
    
    f = compute_I_f(elements, nodes)
    b = M.dot(f)
    
    u = solve(A, b, boundary_nodes)
    fig = plot_solution(nodes[:, 0], nodes[:,1], u)
    
    plt.show()
