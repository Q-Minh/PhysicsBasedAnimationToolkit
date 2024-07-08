# The Finite Element Method

## Introduction

The Finite Element Method (FEM) is an approach to discretizing space-dependent continuous problems defined on geometrically "complex" domains. As opposed to Finite Differences (FD), which is restricted to regular grid domains, FEM relies on [meshing](https://en.wikipedia.org/wiki/Mesh_generation) the problem's domain (geometric discretization), describing the problem's solution as a linear combination of piece-wise (i.e. per mesh element) interpolating polynomials (functional discretization), and solving the approximated problem on the mesh. The "Finite" in FEM comes from limiting the approximate solution to be spanned by a "finite-dimensional" functional basis, i.e. the one spanned by the element-wise interpolating polynomials (the basis functions). The "Element" in FEM comes from the polynomials being entirely constructed/defined by the simple geometric primitives composing the mesh, called "elements".

## Prerequisites

- Calculus
- Linear Algebra
- Differential equations (optional)

## Method

Given some space-dependent problem whose solution $u(X)$ is defined on some 
continuous domain $\Omega$, FEM first requires a *geometric mesh* $(V,C) \approx \Omega$, where $C$ is a set of cells whose vertices are in $V$. From $(V,C)$, we then construct the *FEM mesh* $(I,E)$, where $I$ is the set of FEM *nodes* and $E$ is the set of FEM *elements*. We now assume that the geometric mesh is the domain itself $(V,C)=\Omega$.

Nodes $i \in I$ have corresponding positions $X_i \in \Omega$ and associated *basis functions* $\phi_i(X): \Omega \longrightarrow \mathbb{R}$ such that $\phi_i(X_j) = \delta_{ij}$, where $\delta_{ij}$ is the [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta). Such basis functions are defined element-wise by so-called *shape functions* $N_i^e(X)$ for pairs $(i,e)$ of adjacent node and element, and vanish in non-adjacent elements. In other words, the support of $\phi_i(X)$ is the union of its adjacent elements, and evaluating $\phi_i(X)$ amounts to finding the element $e$ containing the evaluation point $X$, and evaluating $N_i^e(X)$ there. It is common to refer to this choice of basis function $\phi_i(X)$ as the "hat" function, because its tip (i.e. its maximal value of $1$) is located on node $i$, "centered" in its support, while it smoothly decreases to 0 at surrounding nodes. It is also common to refer to these elements as "PK Lagrange elements", because their shape functions $N_i^e(X)$ are interpolating polynomials of degree $K$.

| ![Geometric mesh vs FEM mesh](./media/geometric.mesh.vs.fem.mesh.jpg) | 
|:--:| 
| Geometric mesh (left) versus FEM meshes discretized with linear (middle) and quadratic (right) Lagrange elements. |

With $n=|I|$, FEM restricts $u(X)$ to the class of functions $u(X) = \sum_{i=1}^{n} u_i \phi_i(X)$, i.e. linear combinations of coefficients $u_i$ and basis functions $\phi_i$ associated with nodes $i$. More compactly, $u(X) = \Phi^T u$, where $`u = \begin{bmatrix}u_1 & \dots & u_{n} \end{bmatrix}^T \in \mathbb{R}^{n}`$ and $`\Phi = \begin{bmatrix} \phi_1(X) & \dots & \phi_{n}(X) \end{bmatrix}^T \in \mathbb{R}^{n}`$. We say that the basis functions $\phi_i$ span the function space $`\{ \Phi^T u \;\forall\; u \in \mathbb{R}^{n} \}`$, much like basis vectors $v_i$ span vector spaces $V$. Functions in the FEM function space, i.e. the space spanned by $\phi_i$, are uniquely represented by their vector of coefficients $u$, much like vectors $v = v_1 \overrightarrow{i} + v_2\overrightarrow{j} + v_3 \overrightarrow{k}$ in $\mathbb{R}^3$ are uniquely represented by their coefficients $`\begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}`$. 

| ![FEM function space 1D](./media/fem1D.gif) | 
|:--:| 
| FEM function space in 1D using linear "hat" basis functions. |

Such a discrete functional representation allows one to accurately map continuous problems from theory onto computers by needing only to store the discrete and finite-dimensional coefficients $u$ and FEM mesh $(I,E)$, assuming an appropriate geometric meshing of the problem domain is readily available. Furthermore, the linearity of $u(X)$ w.r.t. $\Phi$ naturally translates solving FEM-discretized problems into solving matrix equations.

### Motivation

As a motivating example, consider a well-known partial differential equation (PDE), the Poisson equation, 

$$
\Delta u(X) = f(X) ,
$$

defined on some domain $\Omega$, where $\Delta$ is the [Laplacian](https://en.wikipedia.org/wiki/Laplace_operator#:~:text=In%20mathematics%2C%20the%20Laplace%20operator,scalar%20function%20on%20Euclidean%20space.) and $f(X)$ is some known function in space. In other words, we wish to solve for some function $u(X)$ whose Laplacian is given by $f(X)$ everywhere in $\Omega$ (i.e. $`\;\forall\; X \in \Omega`$). Without further analytical specification on the domain $\Omega$, there is little we can do to solve such a general problem. In fact, most domains that we care about and that appear in real life don't have an analytical description. Even if we did have such a description of the domain, it might still be very hard to solve such a problem by hand. We do, however, have powerful meshing tools which can take human-authored complex geometry as input, and produce meshes that accurately approximate the domain as output.

Assuming we have such a mesh at hand, we can discretize the solution $u(X)$ as described in the previous section, obtaining

$$
\Delta \left[ \sum_j u_j \phi_j(X) \right] = f(X) 
$$

$$
\sum_j u_j \Delta \phi_j(X) = f(X) ,
$$

since only the basis functions $\phi_j(X)$ are space-dependent. Unfortunately, there are now $n$ unknowns for a single equation, which is an under-determined problem. However, we can transform this single equation into $n$ equations by an operator called the Galerkin projection 

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

where $A_{ij} = \int_{\Omega} \phi_i \Delta \phi_j \partial \Omega$ and $f_i = \int_{\Omega} f \phi_i \partial \Omega$. In matrix notation, 

$$
Au=f ,
$$

where $A \in \mathbb{R}^{n \times n}$ and $f \in \mathbb{R}^n$. We can thus solve for $u$ using one of the many available implementations of numerical algorithms for solving linear systems of equations.

> Interestingly, the Galerkin method is essentially a residual minimization procedure in function space, much like it is often possible to minimize the residual $r=Ax-b$ for solving linear systems of equations approximately. Given some functional equation $L(u(X))=0$, we know that approximating $u(X) \approx \Phi^T u$ will yield some error, which manifests as a residual when passing through the functional $L$, i.e. $L(\Phi^T u) = r$. The Galerkin method then requires that $\langle r, \phi_i \rangle = 0$ for each basis function $\phi_i$. In other words, we ask that the residual be orthogonal (i.e. perpendicular) to the function space spanned by $\phi_i$, i.e. any other function $\Phi^T v$ in this space would yield a worse residual than our solution $\Phi^T u$. Similarly, with linear systems, we ask that $r$ be orthogonal to each basis vector of the map $A$, i.e. each column of the matrix $A$. This yields $A^T r = A^T A x - A^T b= 0$, the well known [normal equations](https://en.wikipedia.org/wiki/Ordinary_least_squares#Normal_equations).

#### Summary

This section showed how a hard-to-solve space-dependent continuous problem defined on a complex domain could be solved accurately using existing numerical techniques through FEM, thus motivating its usefulness in practice. The approach used in our example is called Galerkin FEM, which projects the approximated problem onto the function space used to discretize the solution. Petrov-Galerkin methods allow projecting the approximated problem onto a different function space, i.e. there are two sets of basis functions $\phi_i$ and $\psi_i$, for discretizing the solution and the problem, respectively. In some cases, a projection is not required, i.e. the problem itself already reveals the required number of equations.

### Shape functions

Although our motivating example shows how, armed with basis functions $\phi_i$, continuous problems reduce to discrete ones, we have yet to explicitly define such functions. However, we did *implicitly* define them by stating that they must satisfy the following properties
1. $\phi_i$ is a polynomial
2. $\phi_i(X_j) = \delta_{ij}$
3. $`\phi_i(X) = \begin{cases}N_i^e(X) & X \in \Omega^e, \text{node} \;i\; \text{and element} \;e\; \text{are adjacent}\\ 0 & \text{otherwise}\end{cases}`$

The third property introduces $\Omega^e$, which refers to the domain of element $e$, i.e. the space it occupies. Naturally, meshes are made of non-overlapping geometric primitives (i.e. the "elements" $\Omega^e$), connected through their boundaries. This means that evaluating $\phi_i(X)$ is achieved by the following steps.
1. Find which element $e$ contains the point $X$. 
2. Evaluate $N_i^e(X)$.

Fortunately, our shape functions have $C^0$ continuity at element boundaries, meaning $N_i^e(X) = N_i^{e'}(X)$ on the boundary between adjacent elements $e$ and $e'$ (due to uniqueness of interpolating polynomials). Hence, if a point $X$ lies on the boundary between 2 or more elements, we can pick any of these elements in step 1.

We now focus on the secret sauce, the element shape functions $N_i^e(X)$. Properties 1. and 3. state that $\phi_i(X)$ is a polynomial and evaluates to $N_i^e(X)$ on the element $e$. Thus, $N_i^e(X)$ is a polynomial on the element $e$. Polynomials can be written as linear combinations of basis polynomials $P_k(X)$. Suppose that $n^e = |\text{nodes}(e)|$, then if we have $n^e$ such basis polynomials $P_k(X)$, and we have that $N_i^e(X) = \sum_{j \in \text{nodes}(e)} \alpha_{ij} P_j(X) $. More compactly,

$$
N_i^e(X) = P(X)^T \alpha_i ,
$$

where $`P(X) = \begin{bmatrix} P_1(X) & \dots & P_{n^e}(X) \end{bmatrix}^T`$ and $`\alpha_i = \begin{bmatrix} \alpha_{i1} & \dots & \alpha_{in^e} \end{bmatrix}^T`$. Property 2, i.e. the Kronecker delta property, thus translates into $N_i^e(X_j) = \delta_{ij}$ on element $e$. Substituting $N_i^e(X_j)$ for its polynomial expansion in the Kronecker delta property yields

$$
\begin{bmatrix} P(X_1) & \dots & P(X_{n^e}) \end{bmatrix}^T \begin{bmatrix} \alpha_1 & \dots & \alpha_{n^e} \end{bmatrix} = I_{n^e \times n^e} ,
$$

where $I_{n^e \times n^e} \in \mathbb{R}^{n^e \times n^e}$ is the identity matrix. In matrix notation,

$$
P^T \alpha = I_{n^e \times n^e} ,
$$

where we have conveniently numbered the nodes of element $e$ as $l=1,2,\dots,n^e$. This numbering choice is often referred to as the "local" indexing of the nodes of element $e$. The "global" indexing of these same nodes refers to the actual nodal indices $i$ corresponding to local nodal indices $l$.

The polynomial basis generally has a quite simple analytical form. For example, taking the monomial basis in 1D for a quadratic polynomial yields $P_1(X) = 1, P_2(X) = X, P_3(X) = X^2$. A linear monomial basis in 3D yields $P_1(X)=1, P_2(X) = X_1, P_3(X) = X_2, P_4(X)=X_3$. These basis polynomials are thus super easy to evaluate, differentiate or integrate in code. What we really need is to find the coefficients $\alpha_{ij}$ that will finalize our definition of the shape functions $N_i^e(X)$. Fortunately, solving the Kronecker equation above amounts to computing the inverse of the transposed matrix of polynomials $P^T$ 

$$
\alpha = P^{-T} .
$$ 

Armed with each matrix $A$ stored for its corresponding element in an FEM computer program, we can easily evaluate $\phi_i(X)$ by finding an element $e$ containing point $X$, converting global node index $i$ into its corresponding local index $l$, and returning $P(X)^T \alpha_{l}$. Fortunately, these polynomial coefficient matrices $A$ are to be precomputed only once, in parallel, for each element.

Unfortunately, $P^T$ can easily become [ill-conditioned](https://en.wikipedia.org/wiki/Condition_number), which makes its inversion [numerically unstable](https://en.wikipedia.org/wiki/Numerical_stability), especially for higher-order polynomial basis'. This phenomenon depends on the geometry of the mesh elements, i.e. the positions of the nodes $X_i$. Intuitively, ill-conditioning of $P^T$ means that some of its cofficients are really large (in magnitude), and some of them are really small (in magnitude). Taking as an example the 1D quadratic monomial evaluated at some element's node with position $X=1000$, we get that its corresponding row in $P^T$ would be $`\begin{bmatrix}1 & 1000 & 1000000\end{bmatrix}`$. Clearly, this is ill-conditioned. 

To address this issue, it is common in practice to define some map $X(\xi)$ that takes points in some *reference* space to the domain $\Omega$, and its inverse $\xi(X) = X^{-1}(\xi)$ such that we can construct shape functions in the reference space, where the geometry of the elements will yield well-conditioned $P^T$. In fact, this concept leads to defining the so-called *reference elements*. The maps $X(\xi)$ and $\xi(X)$ are then defined per-element, and always map from and to the reference element, respectively. Reference shape functions are subsequently defined on the reference element and constructed only once. Evaluating a basis function $\phi_i(X)$ on element $e$ thus amounts to mapping $X$ to $\xi$ using element $e$'s inverse map $\xi(X)$, and then evaluating the reference shape function associated with node $i$ of element $e$. Mathematically, assuming that $N_l(\xi)$ is the reference shape function for domain node $i$ associated with reference node $l$ on the reference element, we have that

$$
\phi_i(X) = N_l(\xi(X)) .
$$

[Chapter 9.3 of Hans Petter Langtangen's FEM book](https://hplgit.github.io/INF5620/doc/pub/main_fem.pdf#page=81.67) (highly recommend that book) provides the following pedagogical visuals, in which dots represent nodes of a quadratic FEM triangular element.

| ![FEM reference to domain linear map](./media/p2.triangle.affine.map.png) | ![FEM reference to domain non-linear map](./media/p2.triangle.isoparametric.map.png) |
|:--:|:--:|
| Linear map from reference element (local) to domain element (global).  | Non-linear map from reference element (local) to domain element (global). |

#### Lagrange elements

Perhaps the simplest and/or most popular type of reference element is the [Lagrange element](https://doc.comsol.com/5.3/doc/com.comsol.help.comsol/comsol_api_xmesh.40.4.html). This type of element is defined by a polynomial order $p$ used to construct the shape functions. As described above, there must be as many polynomial basis' as nodes for the inverse of $P^T$ to exist, i.e. $P^T$ must be square. Lagrange elements in 3D thus define their nodes with coordinates $`\xi_l \in \left\{ \left(\frac{a}{p}, \frac{b}{p}, \frac{c}{p}\right) \right\}`$
and corresponding polynomial basis functions $`P_l \in \left\{ \xi_x^a \xi_y^b \xi_z^c \right\}`$ for integer values $0 \leq a,b,c \leq p$, where they are used as powers. In 2D, we only use $a$ and $b$. In 1D, we reduce further to simply using $a$. Simplex elements, such as triangles and tetrahedra have the additional constraint $a+b+c \leq p$, wheras line segments, quadrilaterals and hexahedra do not.

Taken once again from [chapter 9.3 of Hans Petter Langtangen's FEM book](https://hplgit.github.io/INF5620/doc/pub/main_fem.pdf#page=81.67), here are examples of Lagrange elements and their nodes in dimensions `1,2,3`.

| ![P1 Lagrange simplex elements](./media/p1.lagrange.simplex.elements.png) | ![P2 Lagrange simplex elements](./media/p2.lagrange.simplex.elements.png) |
|:--:|:--:|
| Linear Lagrange simplex elements.  | Quadratic Lagrange simplex elements. |

#### Mapping from/to reference

Given exact reference node placements $\xi_l$ and the polynomial basis functions $P_l(\xi)$, we can obtain our reference shape functions $N_l(\xi) = P(\xi)^T \alpha_l$. We can recover the map $X(\xi)$ easily by simple interpolation of domain positions $X_i$ of domain nodes and elements, $i$ and $e$, stored on corresponding reference nodes $l$ with positions $\xi_l$ and shape functions $N_l(\xi)$ on the reference element. Mathematically, we write the map as

$$
X(\xi) = X^e N(\xi) ,
$$

where $`X^e = \begin{bmatrix} X_1 & \dots & X_{n^e} \end{bmatrix} \in \mathbb{R}^{d \times n^e}`$ are element $e$'s nodes' positions $X_i$, $`N(\xi) = \begin{bmatrix} N_1(\xi) & \dots & N_{n^e}(\xi) \end{bmatrix}^T \in \mathbb{R}^{n^e}`$ are the reference shape functions evaluated at $\xi$, and $d$ is the number of embedding dimensions for $X_i$. 

The inverse map $\xi(X)$ is, however, not so trivial in the general case. One way to obtain $\xi(X)$ numerically is by solving the non-linear least-squares problem

$$
\min_{\xi} || X - X(\xi) ||_2^2 ,
$$

for which we can use a [Gauss-Newton algorithm](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm). If the map $X(\xi)$ is linear, however, its jacobian $J$ must be constant. We can choose an arbitrary point around which to perform a Taylor expansion of $X(\xi)$, for example, the reference space's origin $\xi_1 = 0$, which we know is mapped to $X_1$ in Lagrange elements, revealing 

$$
X(\xi) = X_1 + J \xi .
$$

If the reference element's dimensions match the domain's embedding dimensions, $J$ is square and

$$
\xi(X) = J^{-1} (X - X_1) .
$$

Otherwise, the normal equations may be used to define it instead

$$
\xi(X) = (J^T J)^{-1} J^T (X - X_1) .
$$

#### Derivatives

Computing derivatives of basis functions $\phi_i(X)$ also amounts to computing derivatives of shape functions $N_i^e(X)$. Because our shape functions are now defined in reference space, we must use the [chain rule of differentiation](https://en.wikipedia.org/wiki/Chain_rule) to compute $\nabla_X N_i^e(X) = \nabla_X N_l(\xi(X))$, such that 

$$
\nabla_X \phi_i(X) = \nabla_X N_i^e(X) = \nabla_\xi N_l(\xi(X)) \nabla_X \xi(X) 
$$

for $X \in \Omega^e$.

The gradient of the reference shape functions with respect to reference positions is easy enough to compute, since we just need to differentiate polynomials $\xi_x^a \xi_y^b \xi_z^c$. The jacobian $\nabla_X \xi(X)$ of the inverse map is, again, not so trivial in the general case. If a Gauss-Newton algorithm is used to compute $\xi(X)$ as described above, we need to compute and accumulate gradients of the Gauss-Newton iterations by chain rule. Once again, though, if the map is linear, we can use the previous derivations of the linear inverse map to get $\nabla_X \xi(X)$ as 

$$
\nabla_X \xi(X) = J^{-1}
$$

for a square jacobian $J$, and 

$$
\nabla_X \xi(X) = (J^T J)^{-1} J^T
$$

for a non-square jacobian $J$.

#### Summary

Basis functions $\phi_i(X)$ are constructed piece-wise in mesh elements $e$ adjacent to node $i$ via element shape functions $N_i^e(X)$. To evaluate $\phi_i(X)$, we find element $e$ adjacent to node $i$ containing point $X$, and evaluate $N_i^e(X)$. The shape functions are polynomials which depend on the geometry of their associated element, and we compute them by inverting a matrix $P^T$ of polynomials evaluated at element $e$'s nodes. To avoid ill-conditioning of the matrix inversion problem, we instead define a reference element that is well-conditioned, and construct reference shape functions $N_l(\xi)$ there. Evaluating our basis functions in the domain then amounts to $\phi_i(X) = N_i^e(X) = N_l(\xi(X))$, where $\xi(X)$ is the inverse map taking domain positions $X$ to reference positions $\xi$, as opposed to the map $X(\xi)$ which takes reference positions to domain positions. Gradients of basis functions can be computed by carrying out the chain rule of differentiation through $N_l(\xi)$ and $\xi(X)$. The specific placement of nodes and the associated basis polynomials in reference space defines the type of an FEM element. We present the classic Lagrange element as a simple yet popular choice.

Because functions discretized on an FEM mesh have a linear combination structure $u(X) \approx \Phi^T u$, and $u$ is a simple vector of constant (with respect to $X$) coefficients, differential operators $D(\cdot)$ applied to $u(X)$ need only be applied to the basis functions $\phi_i(X)$ by linearity. In other words, $L(u(X)) = \sum_i u_i L(\phi_i(X))$. Hence, the gradient $\nabla u(X)$, for example, amounts to $\sum_i u_i \nabla \phi_i(X)$, where we have shown how to evaluate $\nabla \phi_i(X)$. The same applies to other differential operators, such as $\Delta, \nabla^2, \nabla \cdot, \int \partial$, etc.

### Spatial integration

Many of the problems solved by FEM are defined by integration over the domain $\Omega$. For example, we have seen that PDEs can be solved with a Galerkin projection, which involves computing $\int_{\Omega} L(\Phi^T u) \phi_i(X) \partial \Omega$, where $L(\cdot)$ is a PDE. Spatial integration also arises when we wish to minimize some quantity "everywhere" in the domain, a very common scenario. For example, suppose we have a function $h(X)$ which measures temperature in the domain $\Omega$, and suppose that there is a heat source in some region $\Omega_h \subset \Omega$. Maybe we want to minimize the temperature everywhere in $\Omega$, in the presence of such a heat source. Mathematically, we would thus want to minimize the energy $\int_{\Omega} h(X) \partial \Omega$ subject to $h(\Omega_h) = h_D$, where $h_D$ is the temperature of the heat source.

Thanks to the [separability of definite integrals](https://en.wikipedia.org/wiki/Integral#Conventions), integrals over the domain $\Omega$ can be broken up into a sum of integrals over element domains $\Omega^e$, since elements are non-overlapping and cover the domain. In other words, given some integrand $F(X)$, 

$$
\int_{\Omega} F(X) \partial \Omega = \sum_{e \in E} \int_{\Omega^e} F(X) \partial \Omega .
$$

As such, if we know how to compute an element integral, then we know how to compute integrals over the whole domain by summation. However, elements can have many different configurations depending on the problem. Fortunately, we can leverage the method of [integration by substitution](https://en.wikipedia.org/wiki/Integration_by_substitution#Substitution_for_multiple_variables) (i.e. change of variables), our fixed reference element with known bounds, and the map $X(\xi)$ to compute domain element integrals by integrating in the reference element. Mathematically, 

$$
\int_{\Omega^e} F(X) \partial \Omega = \int_{\Omega^\text{ref}} F(X(\xi)) |\det \nabla_\xi X| \partial \Omega^\text{ref} ,
$$

where $\Omega^\text{ref}$ is the reference element's domain. For reference line, quadrilateral and hexahedral elements, the bounds of integration (i.e. the domain of $\Omega^\text{ref}$) are $0 \leq \xi \leq 1$. For a triangle, the bounds are $\xi_x \in [0, 1], \xi_y \in [0,1-\xi_x]$, whereas for tetrahedra, they become $\xi_x \in [0, 1], \xi_y \in [0, 1- \xi_x], \xi_z \in [0, 1 - \xi_y - \xi_x]$. If the map $X(\xi)$ is not square, i.e. the reference element is of lower dimension than the mesh's embedding dimensions, then the determinant $\det \nabla_\xi X$ is undefined. In fact, a more general expression would replace $\det \nabla_\xi$ with $\Pi_{k} \sigma_k$ where $\sigma_k$ are [singular values](https://en.wikipedia.org/wiki/Singular_value_decomposition) of $\nabla_\xi X$. This approach must be used, for example, when using FEM on 3D triangles, where the reference triangle is inherently a 2D object. For the remainder of the text, we will stick to the notation $\det \nabla_\xi X$, although the singular value variant of the expression is to be implicitly understood.

Although it is possible to analytically derive computable expressions for these reference integrals, it is often not practical to do so. A more general approach (and sometimes more efficient) approach is to use numerical integration, also known as [*quadrature*](https://en.wikipedia.org/wiki/Numerical_integration). Quadrature *rules* are pairs of weights $w_g$ and points $\xi_g$ for which an integral can be approximated by simple weighted sum of the integrand, without computing antiderivatives, as

$$
\int_{\Omega^\text{ref}} F(X(\xi)) |\det \nabla_\xi X| \partial \Omega^\text{ref} \approx \sum_g w_g F(X(\xi_g)) |\det \nabla_\xi X| .
$$

Such weights $w_g$ and points $\xi_g$ are often provided in the form of tables by many FEM implementations for common geometries such as the reference line, triangle, quadrilateral, tetrahedral and hexahedral elements. The specific number of pairs $(w_g, \xi_g)$ and their values depends on the geometry and the type of integrand. Generally, quadrature rules for polynomial integrands are easily obtainable and are, in fact, exact up to floating point precision. The higher the order of integrand, the higher the number of required pairs $(w_g, \xi_g)$ to compute the analytic integral exactly. This statement is relevant, because FEM shape functions are, in fact, polynomials, and integrals over FEM functions become integrals over FEM shape functions, thanks to linearity of the integral operator and the linear combination structure of FEM functions. As an example, consider how $\int_{\Omega} \sum_i u_i \phi_i(X) \partial \Omega = \sum_i u_i \int_{\Omega} \phi_i(X) \partial \Omega$, where we need only know how to integrate $\int_{\Omega} \phi_i(X) \partial \Omega$. Many other integral expressions also reduce to integrating simple integrands involving only basis functions $\phi_i(X)$. Thus, such integrands are also polynomials, and can be computed exactly via quadrature.

#### Summary

In FEM, integrals over the domain $\Omega$ are equivalent to the sum of integrals over elements $\Omega^e$. These element integrals, in turn, can be computed in the reference element using the map $X(\xi)$ and the change of variables technique, because reference elements have fixed and known bounds. The key ingredients to implementing integrals on a computer are to first obtain tables of quadrature weights and points $(w_g, \xi_g)$ for the specific element and integrand, the integrand $F(X)$, the map $X(\xi)$ and the determinant of its jacobian $|\det \nabla_\xi X|$. In pseudocode,

```python
def integrate_element(wg, Xig, F, X, detJ):
    I = 0
    for g in range(wg.shape[0]):
        I = I + wg[g] * F(X.map(Xig[:,g])) * detJ[g]
    return I

def integrate_domain(mesh, wg, Xig, F):
    I = 0
    for e in mesh.elements:
        X = mesh.reference_to_domain_map(e)
        detJ = X.jacobian_determinants_at_reference_points(Xig)
        I = I + integrate_element(wg, Xig, F, X, detJ)
    return I
```

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

In matrix notation, this is exactly $Mf$, where $M \in \mathbb{R}^{n \times n}$ and $f \in \mathbb{R}^{n \times n}$ and the entries $M_{ij} = \int_{\Omega} \phi_i(X) \phi_j(X) \partial \Omega$. The forcing function $f(X)$ may thus be user-defined purely by specifying function values $f_i$ at the mesh nodes $i$, rather than at mesh elements $e$ as in the previous section. If $\phi_i(X)$ is a polynomial of order $p$, then mass matrix entries have polynomial integrands of order $2p$. Thus, the mass matrix can be computed exactly using a polynomial quadrature rule of order $2p$.

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

Interestingly, we can discretize the [divergence](https://en.wikipedia.org/wiki/Divergence) operator by simply transposing the gradient matrix, i.e. $D = G^T \in \mathbb{R}^{n \times d|E|q}$. Given some vector field $F(X)$ discretized at element quadrature points $X(\xi_g)$, where each scalar field component $F_k(X)$ of $F(X)$ has been discretized into $F_k \in \mathbb{|E|q \times 1}$ and stacked into $F \in \mathbb{d|E|q \times 1}$, 

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

## Limitations

While FEM remains an invaluable tool for scientific computing, it is not without its drawbacks. One of its main drawbacks is FEM's reliance on an appropriate meshing of the domain. Indeed, for FEM discretized problems to be well-conditioned, its element geometries must be "regular". The well-known paper [What is a good linear finite element?](https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf) details this phenomenon in depth. Current meshing tools may take enormous amounts of computational resources (memory, runtime) to produce acceptable outputs. Additionally, it is not guaranteed that any domain *can* be meshed with current tools. In the realm of computer graphics, geometries are extremely complex, i.e. they exhibit fine details, are high resolution, have highly varying structural qualities (like thin structures, high curvatures) in different regions, and may be poorly authored (among other potential obstacles). In such cases, current tools may break down, or FEM will likely struggle with the quality of their output. Even if high quality meshes of such complex geometries are obtained, their resolution may be so high that solving linear systems of equations arising from FEM will be extremely slow. Modern approaches such as [Monte Carlo Geometry Processing](https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/paper.pdf) and its extensions, [Monte Carlo Fluids](https://riouxld21.github.io/research/publication/2022-mcfluid/) tackle this exact problem. [Meshless methods](https://en.wikipedia.org/wiki/Meshfree_methods) also exist to sidestep the mesh generation preprocess to FEM, among other such alternative approaches that I am simply not aware of. In the particular case of triangle mesh FEM, [intrinsic triangulations](https://nmwsharp.com/media/papers/int-tri-course/int_tri_course.pdf) also enable a high level of robustness to mesh ill-conditioning.

## Workshops

1. [pbatoolkit.fem](./W1.ipynb)
2. [Heat diffusion](./W2.ipynb)
3. [Harmonic solutions](./W3.ipynb)
4. [Least squares and Poisson problems](./W4.ipynb)
5. [Lagrange elements](./W5.ipynb)
6. [Operator assembly](./W6.ipynb)
