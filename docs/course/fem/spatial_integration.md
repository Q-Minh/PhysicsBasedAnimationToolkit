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