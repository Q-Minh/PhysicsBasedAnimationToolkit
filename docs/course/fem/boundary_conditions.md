### Boundary conditions

Neumann boundary conditions are imposed values on the gradient of the problem's solution. These Neumann boundary conditions are often called "natural" boundary conditions, because they are implicitly encoded in the problem (where a laplacian appears) and appear "naturally" when applying Green's identities (see previous subsection), i.e. we can enforce them simply by introducing an extra forcing vector in the discretized linear system.

Dirichlet boundary conditions, i.e. "essential" boundary conditions, are imposed on the problem's solution itself (as opposed to its derivatives) and are necessary to make our PDEs well-determined (i.e. not rank-deficient). It is often the case that we can impose Dirichlet boundary conditions directly on the FEM mesh's nodes $i$, by simply constraining its associated coefficients $u_i = d_i$ for some known value $d_i$. This is the same as saying, in the continuous case, that $u(X_i) = d_i$. Because of the Kronecker delta property on FEM basis functions, $u(X_i) = u_i \phi_i(X_i) = u_i$, and so $u_i = d_i$ for Dirichlet constrained nodes. This approach makes it particularly easy to enforce Dirichlet boundary conditions numerically, as it essentially removes degrees of freedom out of a matrix equation. Consider the linear system $Ax=b$ discretizing some problem via FEM. Assume that the vector $x$ has been partitioned into a vector of unknowns $x_u$ and known values $x_k = d_k$ for Dirichlet imposed values $d_k$. The same partitioning may be applied to the rows and columns of matrix $A$ and similarly to the right-hand side vector $b$. We thus get that 

$$
\begin{bmatrix} 
A_{uu} & A_{uk} \\ 
A_{ku} & A_{kk} 
\end{bmatrix} 
\begin{bmatrix} 
 x_u \\
 x_k 
\end{bmatrix} = 
\begin{bmatrix} 
b_u \\ 
b_k 
\end{bmatrix} .
$$

The unknowns are thus obtained by solving a reduced problem

$$
A_{uu} x_u = b_u - A_{uk} d_k .
$$

The matrix $A_{uu}$ preserves symmetry if $A$ is symmetric, and similarly for positive (or negative) (semi-)definiteness.

In the general case, however, it might be the case that Dirichlet boundary conditions cannot be imposed at the nodes. In this case, we might need to enforce Dirichlet boundary conditions as constraints to an optimization problem. For example, our discretized Poisson problem could become the equality constrained minimization

$$
Au = f \longrightarrow \min_u \frac{1}{2}||Au - f||_2^2 
\quad\text{ s.t. }\quad
Du_D - d_D = 0 ,
$$

where $Du_D - d_D = 0$ discretizes the continuous constraint $\int_{\partial \Omega^D} u(X) \partial S - d(X) = 0$ using FEM.

### Vector-valued functions

In many cases, we might want to discretize vector-valued functions using FEM. For example, we might want to model some displacement field in the domain, as is the case in elasticity/deformation. Suppose that $u(X): \Omega \longrightarrow \mathbb{R}^d$ for some $d$, i.e. $u(X)$ maps domain positions to some $d$-dimensional quantity. In this case, we can simply consider each scalar component $u_k(X)$ of $u(X)$ as a function. Thus, we can discretize each $u_k(X)$ using FEM and obtain $d$ coefficient vectors $u_k \in \mathbb{R}^n$. All the operators that we have already defined can then directly be applied to each $u_k$ independently. In matrix notation, we may horizontally stack the $u_k$ as columns into some matrix $U \in \mathbb{R}^{n \times d}$, and multiply $U$ from the left using operators $M,L,G$. The product $LU$ would compute the Laplacian of each scalar component $u_k(X)$ of $u(X)$, while $GU$ would compute their gradients. The load vector can similarly be defined as considering each $u_k(X)$ as its own separate forcing function, resulting in a matrix $F \in \mathbb{R}^{n \times d}$ after discretization.

In other situations, however we may wish to concatenate the coefficients $u_k$ into a single vector $u \in \mathbb{R}^{nd}$. This leads to 2 options:

1. Stack each $u_k$ vertically.
2. Interleave all the $u_k$.

By interleaving, we mean that if the $k^{\text{th}}$ coefficient vector $u_k$ has coefficients $u_{ik}$, then the resulting "interleaved" vector $`u = \begin{bmatrix} u_{11} & \dots & u_{n1} & \dots & u_{1d} & \dots & u_{nd} \end{bmatrix}^T`$. Practically, this just means that option 1 flattens the matrix $U \in \mathbb{R}^{n \times d}$ by stacking its column vertically, and option 2 flattens $U$ by stacking its transposed rows vertically. For both options, our FEM operators $M,L,G$ can adjust to $u$ by using the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product). Our new operators $A = M, L, G$ take the following forms for the corresponding options 

1. $A_{nd \times nd} = I_{d \times d} \otimes A$ 
2. $A_{nd \times nd} = A \otimes I_{d \times d}$ 

The new load vector $f \in \mathbb{R}^{nd}$ can be obtained from $F$ in the same way as $u$ was obtained from $U$.

In the continuous setting, FEM discretized functions (i.e. solution, forcing function, etc.) are now written in the following equivalent forms

$$
u(X) = \sum_i \left[I_{d \times d} \otimes \phi_i(X)\right] u_i = \sum_i \phi_i(X) u_i,
$$

and their gradients/jacobians become 

$$
\nabla u(X) = \sum_i u_i \nabla \phi_i(X) ,
$$

where $u_i$ is a column vector, while the gradient $\nabla \phi_i(X)$ is a row vector, i.e. $u_i \nabla \phi_i(X)$ is their outer product.