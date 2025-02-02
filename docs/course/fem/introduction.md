# The Finite Element Method

## Introduction

The Finite Element Method (FEM) is an approach to discretizing space-dependent continuous problems defined on geometrically "complex" domains. As opposed to Finite Differences (FD), which is restricted to regular grid domains, FEM relies on [meshing](https://en.wikipedia.org/wiki/Mesh_generation) the problem's domain (geometric discretization), describing the problem's solution as a linear combination of piece-wise (i.e., per mesh element) interpolating polynomials (functional discretization), and solving the approximated problem on the mesh. The "Finite" in FEM comes from limiting the approximate solution to be spanned by a "finite-dimensional" functional basis, i.e., the one spanned by the element-wise interpolating polynomials (the basis functions). The "Element" in FEM comes from the polynomials being entirely constructed/defined by the simple geometric primitives composing the mesh, called "elements".

<p>
  <img src="../../../_static/imgs/geometric.mesh.vs.fem.mesh.jpg" alt="Geometric mesh vs FEM mesh with linear and quadratic Lagrange elements" width="500" />
</p>
<p style="text-align:center;">Geometric mesh (left) vs FEM mesh with linear (middle) and quadratic (right) Lagrange elements.</p>


With $n=|I|$, FEM restricts $u(X)$ to the class of functions $u(X) = \sum_{i=1}^{n} u_i \phi_i(X)$, i.e., linear combinations of coefficients $u_i$ and basis functions $\phi_i$ associated with nodes $i$. More compactly, $u(X) = \Phi^T u$, where $u = \begin{bmatrix}u_1 & \dots & u_{n} \end{bmatrix}^T \in \mathbb{R}^{n}$ and $\Phi = \begin{bmatrix} \phi_1(X) & \dots & \phi_{n}(X) \end{bmatrix}^T \in \mathbb{R}^{n}$. We say that the basis functions $\phi_i$ span the function space $\{ \Phi^T u \;\forall\; u \in \mathbb{R}^{n} \}$, much like basis vectors $v_i$ span vector spaces $V$. Functions in the FEM function space, i.e., the space spanned by $\phi_i$, are uniquely represented by their vector of coefficients $u$, much like vectors $v = v_1 \overrightarrow{i} + v_2\overrightarrow{j} + v_3 \overrightarrow{k}$ in $\mathbb{R}^3$ are uniquely represented by their coefficients $\begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}$. 

<p>
  <img src="../../../_static/imgs/fem1D.gif" alt="FEM function space in 1D using linear hat basis functions" width="500" />
</p>
<p style="text-align:center;">FEM function space in 1D using linear hat basis functions.</p>


Such a discrete functional representation allows one to accurately map continuous problems from theory onto computers by needing only to store the discrete and finite-dimensional coefficients $u$ and FEM mesh $(I,E)$, assuming an appropriate geometric meshing of the problem domain is readily available. Furthermore, the linearity of $u(X)$ w.r.t. $\Phi$ naturally translates solving FEM-discretized problems into solving matrix equations.


## Prerequisites

- Calculus
- Linear Algebra
- Differential equations (optional)
