The Finite Element Method
=========================

## Introduction

The Finite Element Method (FEM) is an approach to discretizing space-dependent continuous problems defined on geometrically "complex" domains. As opposed to Finite Differences (FD), which is restricted to regular grid domains, FEM relies on `meshing <https://en.wikipedia.org/wiki/Mesh_generation>`_ the problem's domain (geometric discretization), describing the problem's solution as a linear combination of piece-wise (i.e. per mesh element) interpolating polynomials (functional discretization), and solving the approximated problem on the mesh. The "Finite" in FEM comes from limiting the approximate solution to be spanned by a "finite-dimensional" functional basis, i.e., the one spanned by the element-wise interpolating polynomials (the basis functions). The "Element" in FEM comes from the polynomials being entirely constructed/defined by the simple geometric primitives composing the mesh, called "elements".

## Prerequisites

- Calculus
- Linear Algebra
- Differential equations (optional)

## Method

Given some space-dependent problem whose solution :math:`u(X)` is defined on some 
continuous domain :math:`\Omega`, FEM first requires a *geometric mesh* :math:`(V,C) \approx \Omega`, where :math:`C` is a set of cells whose vertices are in :math:`V`. From :math:`(V,C)`, we then construct the *FEM mesh* :math:`(I,E)`, where :math:`I` is the set of FEM *nodes* and :math:`E` is the set of FEM *elements*. We now assume that the geometric mesh is the domain itself :math:`(V,C)=\Omega`.

Nodes :math:`i \in I` have corresponding positions :math:`X_i \in \Omega` and associated *basis functions* :math:`\phi_i(X): \Omega \longrightarrow \mathbb{R}` such that :math:`\phi_i(X_j) = \delta_{ij}`, where :math:`\delta_{ij}` is the `Kronecker delta <https://en.wikipedia.org/wiki/Kronecker_delta>`_. Such basis functions are defined element-wise by so-called *shape functions* :math:`N_i^e(X)` for pairs :math:`(i,e)` of adjacent node and element, and vanish in non-adjacent elements. In other words, the support of :math:`\phi_i(X)` is the union of its adjacent elements, and evaluating :math:`\phi_i(X)` amounts to finding the element :math:`e` containing the evaluation point :math:`X`, and evaluating :math:`N_i^e(X)` there. It is common to refer to this choice of basis function :math:`\phi_i(X)` as the "hat" function, because its tip (i.e., its maximal value of 1) is located on node :math:`i`, "centered" in its support, while it smoothly decreases to 0 at surrounding nodes. It is also common to refer to these elements as "PK Lagrange elements", because their shape functions :math:`N_i^e(X)` are interpolating polynomials of degree :math:`K`.

.. list-table:: 
   :header-rows: 0
   :widths: auto

   * - .. image:: ./media/geometric.mesh.vs.fem.mesh.jpg
       :alt: Geometric mesh vs FEM mesh 
     - Geometric mesh (left) versus FEM meshes discretized with linear (middle) and quadratic (right) Lagrange elements.

With :math:`n=|I|`, FEM restricts :math:`u(X)` to the class of functions :math:`u(X) = \sum_{i=1}^{n} u_i \phi_i(X)`, i.e., linear combinations of coefficients :math:`u_i` and basis functions :math:`\phi_i` associated with nodes :math:`i`. More compactly, :math:`u(X) = \Phi^T u`, where 

.. math::

    u = \begin{bmatrix}u_1 & \dots & u_{n} \end{bmatrix}^T \in \mathbb{R}^{n}
    \Phi = \begin{bmatrix} \phi_1(X) & \dots & \phi_{n}(X) \end{bmatrix}^T \in \mathbb{R}^{n}

We say that the basis functions :math:`\phi_i` span the function space :math:`\{ \Phi^T u \;\forall\; u \in \mathbb{R}^{n} \}`, much like basis vectors :math:`v_i` span vector spaces :math:`V`. Functions in the FEM function space, i.e., the space spanned by :math:`\phi_i`, are uniquely represented by their vector of coefficients :math:`u`, much like vectors :math:`v = v_1 \overrightarrow{i} + v_2\overrightarrow{j} + v_3 \overrightarrow{k}` in :math:`\mathbb{R}^3` are uniquely represented by their coefficients :math:`\begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}`.

.. list-table:: 
   :header-rows: 0
   :widths: auto

   * - .. image:: ./media/fem1D.gif
       :alt: FEM function space 1D
     - FEM function space in 1D using linear "hat" basis functions.

Such a discrete functional representation allows one to accurately map continuous problems from theory onto computers by needing only to store the discrete and finite-dimensional coefficients :math:`u` and FEM mesh :math:`(I,E)`, assuming an appropriate geometric meshing of the problem domain is readily available. Furthermore, the linearity of :math:`u(X)` w.r.t. :math:`\Phi` naturally translates solving FEM-discretized problems into solving matrix equations.

### Motivation

As a motivating example, consider a well-known partial differential equation (PDE), the Poisson equation, 

.. math::

    \Delta u(X) = f(X),

defined on some domain :math:`\Omega`, where :math:`\Delta` is the `Laplacian <https://en.wikipedia.org/wiki/Laplace_operator#:~:text=In%20mathematics%2C%20the%20Laplace%20operator,scalar%20function%20on%20Euclidean%20space.>`_ and :math:`f(X)` is some known function in space. In other words, we wish to solve for some function :math:`u(X)` whose Laplacian is given by :math:`f(X)` everywhere in :math:`\Omega` (i.e., :math:`\;\forall\; X \in \Omega`). Without further analytical specification on the domain :math:`\Omega`, there is little we can do to solve such a general problem. In fact, most domains that we care about and that appear in real life don't have an analytical description. Even if we did have such a description of the domain, it might still be very hard to solve such a problem by hand. We do, however, have powerful meshing tools which can take human-authored complex geometry as input, and produce meshes that accurately approximate the domain as output.

Assuming we have such a mesh at hand, we can discretize the solution :math:`u(X)` as described in the previous section, obtaining

.. math::

    \Delta \left[ \sum_j u_j \phi_j(X) \right] = f(X) 

.. math::

    \sum_j u_j \Delta \phi_j(X) = f(X) ,

since only the basis functions :math:`\phi_j(X)` are space-dependent. Unfortunately, there are now :math:`n` unknowns for a single equation, which is an under-determined problem. However, we can transform this single equation into :math:`n` equations by an operator called the Galerkin projection 

.. math::

    \int_{\Omega} \sum_j u_j \Delta \phi_j(X) \phi_i(X) \partial \Omega = \int_{\Omega} f(X) \phi_i(X) \partial \Omega .

Much like the dot product :math:`\langle a, b \rangle = a^T b` for vectors :math:`a` and :math:`b`, :math:`\langle f, g \rangle = \int_{\Omega} f(X)g(X) \partial \Omega` for functions :math:`f` and :math:`g`. In this sense, much like we say that :math:`a^T b` projects the vector :math:`a` onto the vector :math:`b`, then :math:`\int_{\Omega} f(X) g(X) \partial \Omega` projects the function :math:`f` onto the function :math:`g`. Thus, omitting the dependence on :math:`X` to simplify notation, such a projection reveals the :math:`n` equations

.. math::

    \sum_j u_j \langle \phi_i, \Delta \phi_j \rangle = \langle f, \phi_i \rangle

for :math:`i=1,2,\dots,n`, matching the :math:`n` unknowns :math:`u_j` for :math:`j=1,2,\dots,n`. The FEM's functional discretization of :math:`u` and the Galerkin projection thus yield a linear system of equations 

.. math::

    \sum_j A_{ij} u_j = f_i

where :math:`A_{ij} = \int_{\Omega} \phi_i \Delta \phi_j \partial \Omega` and :math:`f_i = \int_{\Omega} f \phi_i \partial \Omega`. In matrix notation, 

.. math::

    Au=f ,

where :math:`A \in \mathbb{R}^{n \times n}` and :math:`f \in \mathbb{R}^n`. We can thus solve for :math:`u` using one of the many available implementations of numerical algorithms for solving linear systems of equations.

.. tip::
   Interestingly, the Galerkin method is essentially a residual minimization procedure in function space, much like it is often possible to minimize the residual :math:`r=Ax-b` for solving linear systems of equations approximately. Given some functional equation :math:`L(u(X))=0`, we know that approximating :math:`u(X) \approx \Phi^T u` will yield some error, which manifests as a residual when passing through the functional :math:`L`, i.e., :math:`L(\Phi^T u) = r`. The Galerkin method then requires that :math:`\langle r, \phi_i \rangle = 0` for each basis function :math:`\phi_i`. In other words, we ask that the residual be orthogonal (i.e., perpendicular) to the function space spanned by :math:`\phi_i`, i.e., any other function :math:`\Phi^T v` in this space would yield a worse residual than our solution :math:`\Phi^T u`. Similarly, with linear systems, we ask that :math:`r` be orthogonal to each basis vector of the map :math:`A`, i.e., each column of the matrix :math:`A`. This yields :math:`A^T r = A^T A x - A^T b= 0`, the well-known `normal equations <https://en.wikipedia.org/wiki/Ordinary_least_squares#Normal_equations>`_.

Summary
-------

This section showed how a hard-to-solve space-dependent continuous problem defined on a complex domain could be solved accurately using existing numerical techniques through FEM, thus motivating its usefulness in practice. The approach used in our example is called Galerkin FEM, which projects the approximated problem onto the function space used to discretize the solution. Petrov-Galerkin methods allow projecting the approximated problem onto a different function space, i.e., there are two sets of basis functions :math:`\phi_i` and :math:`\psi_i`, for discretizing the solution and the problem, respectively. In some cases, a projection is not required, i.e., the problem itself already reveals the required number of equations.
