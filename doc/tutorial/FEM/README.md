# The Finite Element Method

## Introduction

The Finite Element Method (FEM) is an approach to discretizing space-dependent continuous problems defined on geometrically "complex" domains. As opposed to Finite Differences (FD), which is restricted to regular grid domains, FEM relies on [meshing](https://en.wikipedia.org/wiki/Mesh_generation) the problem's domain (geometric discretization), describing the problem's solution as a linear combination of piece-wise (i.e. per mesh element) interpolating polynomials (functional discretization), and solving the approximated problem on the mesh. The "Finite" in FEM comes from limiting the approximate solution to be spanned by a "finite-dimensional" functional basis, i.e. the one spanned by the element-wise interpolating polynomials (the basis functions). The "Element" in FEM comes from the polynomials being entirely constructed/defined by the simple geometric primitives composing the mesh, called "elements".

## Prerequisites

- Calculus
- Linear Algebra
- Differential equations (optional)

## Method

Given some space-dependent problem whose solution $u^{*}(X)$ is defined on some 
continuous domain $\Omega$, FEM first requires a *geometric mesh* $(V,C) \approx \Omega$, where $C$ is a set of cells whose vertices are in $V$. From $(V,C)$, we then construct the *FEM mesh* $(I,E)$, where $I$ is the set of FEM *nodes* and $E$ is the set of FEM *elements*. 

Nodes $i \in I$ have corresponding positions $X_i \in \Omega$ and associated *basis functions* $\phi_i(X): \Omega \longrightarrow \mathbb{R}$ such that $\phi_i(X_j) = \delta_{ij}$, where $\delta_{ij}$ is the [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta). Such basis functions are defined element-wise by so-called *shape functions* $N_i^e(X)$ for pairs $(i,e)$ of adjacent node and element, and vanish in non-adjacent elements. In other words, the support of $\phi_i(X)$ is the union of its adjacent elements, and evaluating $\phi_i(X)$ amounts to finding the element $e$ containing the evaluation point $X$, and evaluating $N_i^e(X)$ there. It is common to refer to this choice of basis function $\phi_i(X)$ as the "hat" function, because its tip (i.e. its maximal value of $1$) is located on node $i$, "centered" in its support, while it smoothly decreases to 0 at surrounding nodes. It is also common to refer to these elements as "PK Lagrange elements", because their shape functions $N_i^e(X)$ are interpolating polynomials of degree $K$.

| ![Geometric mesh vs FEM mesh](./media/geometric.mesh.vs.fem.mesh.jpg) | 
|:--:| 
| Geometric mesh (left) versus FEM meshes discretized with linear (middle) and quadratic (right) Lagrange elements. |

FEM restricts $u(X)$ to the class of functions $u(X) = \sum_{i=1}^{|I|} u_i \phi_i(X)$, i.e. linear combinations of coefficients $u_i$ and basis functions $\phi_i$ associated with nodes $i$. More compactly, $u(X) = \Phi^T u$, where $`u = \begin{bmatrix}u_1 & \dots & u_{|I|} \end{bmatrix}^T \in \mathbb{R}^{|I|}`$ and $`\Phi = \begin{bmatrix} \phi_1(X) & \dots & \phi_{|I|}(X) \end{bmatrix}^T \in \mathbb{R}^{|I|}`$. We say that the basis functions $\phi_i$ span the function space $`\{ \Phi^T u \;\forall\; u \in \mathbb{R}^{|I|} \}`$, much like basis vectors $v_i$ span vector spaces $V$. Functions in the FEM function space, i.e. the space spanned by $\phi_i$, are uniquely represented by their vector of coefficients $u$, much like vectors $v = v_1 \overrightarrow{i} + v_2\overrightarrow{j} + v_3 \overrightarrow{k}$ in $\mathbb{R}^3$ are uniquely represented by their coefficients $`\begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}`$. 

| ![FEM function space 1D](./media/fem1D.gif) | 
|:--:| 
| FEM function space in 1D using linear "hat" basis functions. |

Such a discrete functional representation allows one to accurately map continuous problems from theory onto computers by needing only to store the discrete and finite-dimensional coefficients $u$ and FEM mesh $(I,E)$, assuming an appropriate geometric meshing of the problem domain is readily available. Furthermore, the linearity of $u(X)$ w.r.t. $\Phi$ naturally translates solving FEM-discretized problems into solving matrix equations.

### Motivation

As a motivating example, consider a well-known partial differential equation (PDE), the Poisson equation, 
$$
\Delta u(X) = f(X) ,
$$
defined on some domain $\Omega$, where $\Delta$ is the [Laplacian](https://en.wikipedia.org/wiki/Laplace_operator#:~:text=In%20mathematics%2C%20the%20Laplace%20operator,scalar%20function%20on%20Euclidean%20space.) and $f(X)$ is some known function in space. In other words, we wish to solve for some function $u(X)$ whose Laplacian is given by $f(X)$ everywhere in $\Omega$ (i.e. $`\;\forall\; X \in \Omega`$). Without further analytical specification on the domain $\Omega$, there is little we can do to solve such a general problem. In fact, most domains that we care about and that appear in real life don't have an analytical description. Even if we did have such a description of the domain, it might still be very hard to solve such a problem by hand. We do, however, have powerful meshing tools which can take human-authored complex geometry as input, and produce meshes accurately approximating the domain as output.

Assuming we have such a mesh at hand, we can discretize the solution $u(X)$ as described in the previous section, obtaining
$$
\Delta \left[ \sum_j u_j \phi_j(X) \right] = f(X) \\
\sum_j u_j \Delta \phi_j(X) = f(X) ,
$$
since only the basis functions $\phi_j(X)$ are space-dependent. Unfortunately, there are now $n=|I|$ unknowns for a single equation, which is an under-determined problem. However, we can transform this single equation into $n$ equations by an operator called the Galerkin projection 
$$
\int_{\Omega} \sum_j u_j \Delta \phi_j(X) \phi_i(X) \partial \Omega = \int_{\Omega} f(X) \phi_i(X) \partial \Omega .
$$
Much like the dot product $\langle a, b \rangle = a^T b$ for vectors $a$ and $b$, $\langle f, g \rangle = \int_{\Omega} f(X)g(X) \partial \Omega$ for functions $f$ and $g$. In this sense, much like we say that $a^T b$ projects the vector $a$ onto the vector $b$, then $\int_{\Omega} f(X) g(X) \partial \Omega$ projects the function $f$ onto the function $g$. Thus, omitting the dependence on $X$ to simplify notation, such a projection reveals the $n$ equations
$$
\sum_j u_j \langle \phi_i, \Delta \phi_j \rangle = \langle f, \phi_i \rangle
$$
for $i=1,2,\dots,n$, matching the $n$ unknowns $u_j$ for $j=1,2,\dots,n$. The FEM's functional discretization of $u$ and the Galerkin projection thus yield a linear system of equations 
$$
\sum_j A_{ij} u_j = f_i
$$
where $A_{ij} = \int_{\Omega} \phi_i \Delta \phi_j \partial \Omega$ and $f_i = \int_{\Omega} f \phi_i \partial \Omega$. Thus, in matrix notation, 
$$
Au=f ,
$$
where $A \in \mathbb{R}^{n \times n}$ and $f \in \mathbb{R}^n$. We can thus solve for $u$ using one of the many available implementations of numerical algorithms for solving linear systems of equations.


### Shape functions

### Spatial integration

### Operators

### Boundary conditions

### Solving

## Limitations


