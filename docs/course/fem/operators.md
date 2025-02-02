### Operators

Now that we know how to compute shape functions, their derivatives and their integrals, we present a handful of useful FEM operators used to discretize common problems.

#### Shape function matrix

Our first operator will simply evaluate an FEM function $u(X)$ at every quadrature point $X(\xi_g)$ in every element $e$, where $X(\cdot)$ is to be understood as element $e$'s map from the reference element to the domain. Given $u(\cdot)$'s coefficient vector $u$, we thus wish to compute $u(X(\xi_g)) = \sum_i u_i \phi(X(\xi_g)) = \Phi(X(\xi_g))^T u$ for every element $e$ and every quadrature point $\xi_g$ in the reference element. However, we know that $\phi_i(X(\xi_g))$ is only non-zero when node $i$ is part of element $e$, so we can safely ignore (i.e. set to zero) all $\phi_j(X(\xi_g))$ for nodes $j$ not in element $e$. In other words, only the shape functions of element $e$ need to be evaluated at quadrature points in $e$. This results in a highly sparse operator $N$, whose (sparse) block rows $N_e$ are defined as 

$$
N_e = \begin{bmatrix}
\phi_1(X(\xi_1)) & \dots & \phi_n(X(\xi_1)) \\
\vdots & \vdots & \vdots \\
\phi_1(X(\xi_q)) & \dots & \phi_n(X(\xi_q)) \\
\end{bmatrix} \in \mathbb{R}^{q \times n} ,
$$

where $q$ is the number of quadrature points and $n$ is the number of FEM nodes. Compact storage of $N_e$ would store only the shape function values $`\begin{bmatrix} N_l^e(X(\xi_g)) \end{bmatrix} \in \mathbb{R}^{q \times n^e}`$ in a dense matrix using local node indexing $l$. These local dense matrices are often named using the prefix "element", as in "element shape function matrix", or "element hessian" and so on and so forth.

Our full shape function matrix $N \in \mathbb{R}^{|E|q \times n}$ is thus sparse, and its application to $u$ 
computes $u(X^e(\xi_g))$, yielding a vector $N u \in \mathbb{R}^{|E|q \times 1}$.

#### Quadrature matrix

We introduce a second operator that will enable computing integrals of FEM quantities using matrix operations. Recall from the section on spatial integration that using numerical quadrature, any integral in any element $e$ with polynomial integrands of order $p$ can be computed exactly given the quadrature weights $w_g$ and the jacobian determinants $|\det \nabla_\xi X(\xi_g)|$ for a polynomial quadrature rule $(w_g, \xi_g)$ of order greater than or equal to $p$. Hence, any integrand evaluated at $X(\xi_g)$ simply needs to be multiplied by $w_g |\det \nabla_\xi X(\xi_g)|$ to be integrated. This hints at the diagonal matrix $Q \in \mathbb{R}^{|E|q \times |E|q}$, which we will name the "Quadrature" matrix, whose diagonal blocks are 

$$
Q_e = \begin{bmatrix}
w_1 |\det \nabla_\xi X(\xi_1)| & 0 & 0 \\
0 & \ddots & 0 \\
0 & 0 & w_q |\det \nabla_\xi X(\xi_q)| \\
\end{bmatrix} \in \mathbb{R}^{q \times q} 
$$

for every element $e$.

This quadrature matrix $Q$ is essentially a discrete analog of the inner product $\langle u, v \rangle = \int_\Omega u v \partial \Omega$ of functions $u, v$ defined on the FEM mesh. For instance, the volume of each domain can be computed easily by the inner product of the unit function, i.e. $\langle 1, 1 \rangle = \int_\Omega \partial \Omega = 1_{|E|q}^T Q 1_{|E|q}$, where we have used the vector of all ones $1_{|E|q} \in \mathbb{R}^{|E|q}$. The volume of individual elements $e$ can similarly be computed as $\left[ 
I_{|E| \times |E|} \otimes 1_{q} \right]^T Q \left[ 
I_{|E| \times |E|} \otimes 1_{q} \right] $, where $\otimes$ is the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).

Galerkin projections $\langle f, \phi \rangle$ also become easy to compute. Consider some function $f$ discretized at element quadrature points into matrix $F$, then we can compute its Galerkin projection as $N^T Q F$. 

#### Load vector

In the motivating example, we showed how the Poisson equation $\Delta u(X) = f(X)$ became a linear system $Au=f$ after FEM discretization, where $f_i = \int_{\Omega} f(X) \phi_i(X) \partial \Omega$. However, this form is impractical, because for every different forcing function $f(X)$, a different integral expression would have to be generated to compute $f$. Instead, we will often discretize "known" forcing functions, i.e. *load vectors*, as these piece-wise (per element) polynomial functions $f(X) = f_e(X) \in \mathbb{R}^{d}$ for $X \in \Omega^e$. This allows us to compute the *load vector* $f$ as

$$
f_i = \int_{\Omega} f(X) \phi_i(X) \partial \Omega = \sum_e \int_{\Omega^e} f_e(X) \phi_i(X) \partial \Omega^e ,
$$

since the $f_e(X)$ are polynomial in their corresponding element. If $f_e(X)$ has order $p_f$ and $\phi_i(X)$ is of order $p_\phi$, then a polynomial quadrature rule of order $p=p_f + p_\phi$ is exact, and we obtain 

$$
\int_{\Omega^e} f_e(X) \phi_i(X) \partial \Omega^e = \sum_g \left[ w_g |\det \nabla_\xi X(\xi_g)| \right] f_e(X(\xi_g)) \phi_i(X(\xi_g)) .
$$

We can thus construct the matrix $F \in \mathbb{R}^{|E|q \times d}$, whose block rows $F_e \in \mathbb{R}^{q \times d}$ contain values of the forcing function $f_e$ evaluated at $e$'s quadrature points $X(\xi_g)$, and compute the load vector $f$ under Galerkin projection as 

$$
f = N^T Q F \in \mathbb{R}^{n \times d} 
$$

with exact integration, given a quadrature matrix $Q$ discretized for order $p$ integrands.

In the special case that $f(X)$ is piece-wise constant, the element integrals become $f_e(X) \int_{\Omega^e} \phi_i(X) \partial \Omega^e$ in each element, such that $\int_{\Omega^e} \phi_i(X) \partial \Omega^e$ can instead be precomputed and re-used for any new piece-wise constant forcing function $f(X) = f_e(X)$. In this case, $p = p_\phi$, and basis function integrals $\int_{\Omega^e} \phi_i(X) \partial \Omega^e$ are precomputed into $B = N^T Q 1_{|E|q}$. Then, different load vectors can be computed for any piece-wise constant function $f(X) = f_e(X)$ discretized in $F$ for a quadrature order $p$ as 

$$
f = B F \in \mathbb{R}^{n \times d} .
$$

#### Mass matrix

Another approach is to directly discretize the forcing function $f(X) \approx \sum_j f_j \phi_j(X)$ in the same function space as the FEM solution. Under the Galerkin projection, we would thus get 

$$
\int_{\Omega} f(X) \partial \Omega = \sum_j f_j \int_{\Omega} \phi_i(X) \phi_j(X) \partial \Omega .
$$

In matrix notation, this is exactly $Mf$, where $M \in \mathbb{R}^{n \times n}$ and $f \in \mathbb{R}^{n \times 1}$ and the entries $M_{ij} = \int_{\Omega} \phi_i(X) \phi_j(X) \partial \Omega$. The forcing function $f(X)$ may thus be user-defined purely by specifying function values $f_i$ at the mesh nodes $i$, rather than at mesh elements $e$ as in the previous section. If $\phi_i(X)$ is a polynomial of order $p$, then mass matrix entries have polynomial integrands of order $2p$. Thus, the mass matrix can be computed exactly using a polynomial quadrature rule of order $2p$.

Using this approach, the earlier Poisson problem would be discretized into $Au = Mf$. Many other PDEs involving known functions make the mass matrix appear. The dot product $u(X)^T v(X)$ of 2 functions $u(X) = \sum_i u_i \phi_i(X)$ and $v(X) = \sum_i v_i \phi_i(X)$ discretized in FEM's solution space, with coefficients vectors $u$ and $v$, will similarly make the mass matrix appear as $u(X)^T v(X) = u^T M v$. We can thus think of the mass matrix as a suitable inner product matrix which enjoys the desirable property of being symmetric and positive definite.

Again, because each element uniquely contributes to its nodes' basis function integrals, we can compute and store per-element mass matrices independently as 

$$
M^e = 
\begin{bmatrix} 
\int_{\Omega^e} \phi_1 \phi_1 \partial \Omega^e & \dots & \int_{\Omega^e} \phi_1 \phi_{n^e} \partial \Omega^e \\ 
\vdots &  & \vdots \\ 
\int_{\Omega^e} \phi_{n^e} \phi_1 \partial \Omega^e & \dots & \int_{\Omega^e} \phi_{n^e} \phi_{n^e} \partial \Omega^e 
\end{bmatrix} ,
$$

using local node indices $l=1,2,\dots,n^e$ which are then accumulated (i.e. summed) into the global mass matrix $M$ by mapping local indices $l$ into corresponding global indices $i$ and $j$ for rows and columns. We call these the element mass matrices.

It is possible to associate physical meaning to the mass matrix by injecting into its integral form a measure of mass density $\rho$ (i.e. grams per unit volume). If the mass density is specified as piece-wise constants (per element) $\rho_e$, then we simply scale each element mass matrix as $\rho_e M^e$ and sum the scaled element mass matrices into the global mass matrix $M$.

If we use our previously defined quadrature matrix $Q$ and shape function matrix $N$ of quadrature order $2p$, then we can simply write

$$
M = N^T Q N \in \mathbb{R}^{n \times n} .
$$

With this construction, the inner product structure of $M$ is even more obvious. The product $u^T M v = u^T N^T Q N v$ immediately reveals that we are integrating $\langle u, v \rangle$ for functions $u, v$ discretized on the mesh using shape functions $N$.

#### Gradient matrix

Computing the gradient $\nabla u(X)$ amounts to simply $\sum_i u_i \nabla \phi_i(X)$. This leads to defining $G$ similarly to $N$, except we now inject the shape function derivatives $\frac{\partial \phi_i(X(\xi_g))}{\partial X_k}$ for the $k^\text{th}$ dimension, evaluated at element $e$'s quadrature points $X(\xi_g)$, in the $e^{\text{th}}$ block row.

This leads to defining the Galerkin gradient matrix for $X$ of dimensions $d$ via its block rows, as 

$$
G^k_{e} = \begin{bmatrix}
\frac{\partial \phi_1(X(\xi_1))}{\partial X_k} & \dots & \frac{\partial \phi_n(X(\xi_1))}{\partial X_k} \\
\vdots & \ddots & \vdots\\
\frac{\partial \phi_1(X(\xi_q))}{\partial X_k} & \dots & \frac{\partial \phi_n(X(\xi_q))}{\partial X_k} \\
\end{bmatrix} \in \mathbb{R}^{q \times n}.
$$

As in the case of $N$, only the shape function derivatives of element $e$ are non-zero in this block row, such that $G^k_e$ is sparse. We store the gradients separately per dimension, as 

$$
G^k = \begin{bmatrix}
G^k_1 \\ 
\vdots \\ 
G^k_{|E|} \\
\end{bmatrix} \in \mathbb{R}^{|E|q \times n} ,
$$

and then stack them vertically into 

$$
G = \begin{bmatrix}
G^1 \\
\vdots \\
G^d
\end{bmatrix} \in \mathbb{R}^{d|E|q \times n} .
$$ 

The application of $G$ onto $u$ thus computes each spatial derivative of $u$ at element quadrature points. Each $G^k u$ computes the $k^\text{th}$ spatial derivative of $u(X)$ at all element quadrature points, and each $G^k_e u$ computes its $k^\text{th}$ spatial derivative at element $e$'s quadrature points.

If the gradient appears in the problem (i.e. PDE or other) itself, and the problem has been closed under Galerkin projection, we must now compute the "Galerkin" version of the gradient, i.e. $\int_{\Omega} \nabla u(X) \phi_i(X) \partial \Omega = \langle \nabla u, \phi_i \rangle$. By approximation and linearity of the gradient operator, such an expression becomes 

$$
\int_{\Omega} \nabla u(X) \phi_i(X) \partial \Omega = \sum_j u_j \int_{\Omega} \phi_i(X) \nabla \phi_j(X) \partial \Omega .
$$

If $\phi_i$ has order $p$, then $\phi_i \nabla \phi_j$ has order $2p - 1$. Thus, armed with the quadrature matrix $Q$ and shape function matrix $N$ of quadrature order $2p - 1$, 

$$
\left[ I_{d \times d} \otimes N^T Q \right] G u = \bar{G} u
$$

computes the gradient in the Galerkin sense, where $I_{d \times d}$ is the identity matrix and $\bar{G}$ is the Galerkin projected gradient operator. Its entries are 

$$
\bar{G}^k_{ij} = \int_{\Omega} \phi_i(X) \frac{\partial \phi_j(X)}{\partial X_k} \partial \Omega 
$$

for $k=1,2,\dots,d$. 

#### Laplacian matrix

The Poisson problem discretized the Laplacian matrix into $A$ where $A_{ij} = \int_{\Omega} \phi_i(X) \Delta \phi_j(X) \partial \Omega$. However, this results in a non-symmetric matrix, and requires shape functions of order $p \geq 2$, meaning we wouldn't be able to use linear shape functions to solve a problem involving the Laplacian of the solution. Thus, in practice, we will make use of multivariable integration by parts, i.e. [Green's identities](https://en.wikipedia.org/wiki/Green%27s_identities), to transform $\Delta u(X)$ into

$$
\sum_j u_j \int_{\Omega} \phi_i(X) \Delta \phi_j(X) \partial \Omega = \sum_j u_j 
\left[
\int_{\Omega} -\nabla \phi_i(X) \cdot \nabla \phi_j(X) \partial \Omega + 
\int_{\partial \Omega} \phi_i(X) \nabla \phi_j(X) \cdot n \partial S
\right] .
$$

$$
Au = Lu + Nu
$$

Here, $\int_{\partial \Omega} \partial S$ is to be interpreted as a boundary integral (over $\Omega$'s boundary) with $n$ being the boundary's normal. The integration by parts reveals the symmetric Laplacian matrix $L \in \mathbb{R}^{n \times n}$, and the Neumann matrix $N \in \mathbb{R}^{n \times n}$. In fact, $L$ is desirably both symmetric and negative semi-definite (with only rank-1 deficiency). If Neumann boundary conditions have been specified as a known function 

$$
g(X) = \nabla u(X) \cdot n 
$$ 

on the domain's boundary, i.e. $X \in \partial \Omega$, then we replace $Nu$ by $g \in \mathbb{R}^{n \times n}$ such that $g_i = \int_{\partial \Omega} g(X) \phi_i(X) \partial S$. Because this integral is defined over the boundary of the FEM meshing of $\Omega$, we can similarly extract the boundary mesh for $\partial \Omega$, preserving node indexing. We can then discretize $g(X)$ on this boundary mesh using FEM once again as either a load vector if $g(X)$ is defined on boundary faces, or as $Mg$ using the boundary mesh's mass matrix $M$ and coefficients $g$ defined on boundary vertices. There are many cases where the Neumann boundary conditions are even simpler, however, i.e. when $g(X) = \nabla u(X) \cdot n = 0$, in which case the Laplacian matrix is exactly $L$. The resulting Poisson problem would then be $Lu=f$, which is a symmetric negative semi-definite linear system which can be solved efficiently.

The matrix $L$ is quite famous and is equivalent to the so-called ["cotangent Laplacian"](https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Mesh_Laplacians) or the "Laplace-Beltrami" operator mentioned in the literature, for the case of triangular meshes embedded in 3D. Instead of the general derivation presented in this document, when assuming linear shape functions, it is possible to derive quite elegant analytic expressions involving the cotangent function to compute its entries. In our Physics Based Animation Toolkit, we also like to refer to $L$ as the symmetric part of the Laplacian matrix. If $\phi_i(X)$ is of polynomial order $p$, then $L$'s integrands are polynomials of order $2(p-1)$. A polynomial quadrature rule of order $2(p-1)$ is thus sufficient for exact computation of $L$.

The element Laplacian matrices are

$$
L^e = 
-\begin{bmatrix} 
\int_{\Omega^e} \nabla \phi_1 \nabla \phi_1 \partial \Omega^e & \dots & \int_{\Omega^e} \nabla \phi_1 \nabla \phi_{n^e} \partial \Omega^e \\ 
\vdots &  & \vdots \\ 
\int_{\Omega^e} \nabla \phi_{n^e} \nabla \phi_1 \partial \Omega^e & \dots & \int_{\Omega^e} \nabla \phi_{n^e} \nabla \phi_{n^e} \partial \Omega^e 
\end{bmatrix} ,
$$

where local indexing $l =1,2,\dots,n^e$ has been used.

Once again, we can reuse our previously defined operators to construct the Laplacian instead. Using the matrices $Q$ and $G$ of appropriate quadrature order, we get that

$$
L = -G^T \left[ I_{d \times d} \otimes Q \right] G \in \mathbb{R}^{n \times n} .
$$

#### Divergence matrix

Interestingly, we can discretize the [divergence](https://en.wikipedia.org/wiki/Divergence) operator by simply transposing the gradient matrix, i.e. $D = G^T \in \mathbb{R}^{n \times d|E|q}$. Given some vector field $F(X)$ discretized at element quadrature points $X(\xi_g)$, where each scalar field component $F_k(X)$ of $F(X)$ has been discretized into $F_k \in \mathbb{R}^{|E|q \times 1}$ and stacked into $F \in \mathbb{R}^{d|E|q \times 1}$, 

$$
DF = G^T F \in \mathbb{R}^{n \times 1}
$$

computes its divergence. However, in the previous construction of the Laplacian, or more precisely, the symmetric part of the Galerkin projected Laplacian, we had that 

$$
L = -G^T \left[ I_{d \times d} \otimes Q \right] G = \left[ -D \left(I_{d \times d} \otimes Q \right) \right] G ,
$$

where the divergence operator $D$ must act on the gradient $G$ of an FEM function through integration. As such, when solving problems involving divergence, such as least-squares gradient matching problems, i.e. the Poisson problem

$$
\min_u ||\nabla u - F||_2^2 \leftrightarrow \Delta u = \nabla \cdot F ,
$$

the divergence operator should act on the target vector field $F(X)$ in the integrated (i.e. Galerkin) sense. In matrix notation, we discretize the divergence of the target vector field $F(X)$ as $-D \left[ I_{d \times d} \otimes Q \right] F$, for discrete $F$ constructed as in previous examples.