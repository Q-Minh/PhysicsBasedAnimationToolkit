# The Finite Element Method

## Introduction

The Finite Element Method (FEM) is an approach to discretizing space-dependent continuous problems defined on geometrically "complex" domains. As opposed to Finite Differences (FD), which is restricted to regular grid domains, FEM relies on [meshing](https://en.wikipedia.org/wiki/Mesh_generation) the problem's domain (geometric discretization), describing the problem's solution as a linear combination of piece-wise (i.e. per mesh element) interpolating polynomials (functional discretization), and solving the approximated problem on the mesh. The "Finite" in FEM comes from limiting the approximate solution to be spanned by a "finite-dimensional" functional basis, i.e. the one spanned by the element-wise interpolating polynomials (the basis functions). The "Element" in FEM comes from the polynomials being entirely constructed/defined by the simple geometric primitives composing the mesh, called "elements".

## Prerequisites

- Calculus
- Linear Algebra
- Differential equations

## Method

Given some space-dependent problem whose solution $u^{*}(X)$ is defined on some 
continuous domain $\Omega$, FEM first requires a mesh $(V,C) \approx \Omega$, for some *geometric mesh* $(V,C)$, where $C$ is a set of cells whose vertices are in $V$. From $(V,C)$, we then construct the *FEM mesh* $(I,E)$, where $I$ is the set of FEM *nodes* and $E$ is the set of FEM *elements*. 

Nodes $i \in I$ have corresponding positions $X_i \in \Omega$ and associated *basis functions* $\phi_i(X): \Omega \longrightarrow \mathbb{R}$ such that $\phi_i(X_j) = \delta_{ij}$, where $\delta_{ij}$ is the [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta). Such basis functions are defined element-wise by so-called *shape functions* $N_i^e(X)$ for pairs $(i,e)$ of adjacent node and element, and vanish in non-adjacent elements. In other words, the support of $\phi_i(X)$ is the union of its adjacent elements, and evaluating $\phi_i(X)$ amounts to finding the element $e$ containing the evaluation point $X$, and evaluating $N_i^e(X)$ there. It is common to refer to this choice of basis function $\phi_i(X)$ as the "hat" function, because its tip (i.e. its maximal value of $1$) is located on node $i$, "centered" in its support, while it smoothly decreases to 0 at surrounding nodes. It is also common to refer to these elements as "PK Lagrange elements", because their shape functions $N_i^e(X)$ are interpolating polynomials of degree $K$.

| ![Geometric mesh vs FEM mesh](./media/geometric.mesh.vs.fem.mesh.jpg) | 
|:--:| 
| Geometric mesh (left) versus FEM meshes discretized with linear (middle) and quadratic (right) Lagrange elements |

FEM restricts $u(X)$ to the class of functions $`u(X) = \sum_{i=1}^{|I|} u_i \phi_i(X)`$, i.e. linear combinations of coefficients $u_i$ and basis functions $\phi_i$ associated with nodes $i$. More compactly, $u(X) = u^T \Phi$, where $`u = \begin{bmatrix}u_1 & \dots & u_{|I|} \end{bmatrix}^T \in \mathbb{R}^{|I|}`$ and $`\Phi = \begin{bmatrix} \phi_1(X) & \dots & \phi_{|I|}(X) \end{bmatrix} \in \mathbb{R}^{|I|}`$. We say that the basis functions $\phi_i$ span the function space $`\{ u^T \Phi \;\forall\; u \in \mathbb{R}^{|I|} \}`$, much like vectors $v_i$ span vector spaces $V$. Functions in the FEM function space, i.e. the space spanned by $\phi_i$, are uniquely represented by their vector of coefficients $u$, much like vectors $v = v_1 \overrightarrow{i} + v_2\overrightarrow{j} + v_3 \overrightarrow{k}$ in $\mathbb{R}^3$ are uniquely represented by their coefficients $`\begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}`$. Such a discrete functional representation allows one to accurately map continuous problems from theory onto modern computers by needing only to store the discrete and finite-dimensional coefficients $u$, assuming an appropriate meshing of the problem domain is readily available. Furthermore, the linearity of $u(X)$ w.r.t. $\Phi$ naturally translates solving FEM-discretized problems into solving linear systems of equations.

## Limitations

