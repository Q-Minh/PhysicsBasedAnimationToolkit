import pbatoolkit as pbat
import igl
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import scipy as sp
import argparse
import meshio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Heat geodesics demo",
    )
    parser.add_argument("-i", "--input", help="Path to input tetrahedral or triangle mesh", type=str,
                        dest="input", required=True)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    mesh = None
    if "tetra" in imesh.cells_dict.keys():
        V, C = imesh.points, imesh.cells_dict["tetra"]
        mesh = pbat.fem.Mesh(
            V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    if "triangle" in imesh.cells_dict.keys():
        V, C = imesh.points, imesh.cells_dict["triangle"]
        mesh = pbat.fem.Mesh(
            V.T, C.T, element=pbat.fem.Element.Triangle, order=1)

    V, C = mesh.X.T, mesh.E.T
    F = C
    if mesh.element == pbat.fem.Element.Tetrahedron:
        F = igl.boundary_facets(C)
        F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)

    gamma = [0]
    # Construct Galerkin laplacian, mass and gradient operators
    n = V.shape[0]
    qgM = pbat.fem.inner_product_weights(
        mesh, quadrature_order=2).flatten(order="F")
    QM = sp.sparse.diags_array([qgM], offsets=[0])
    NM = pbat.fem.shape_function_matrix(mesh, quadrature_order=2)
    M = NM.T @ QM @ NM
    GNeL = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    G = pbat.fem.Gradient(
        mesh, GNeL, quadrature_order=1).to_matrix()
    qgL = pbat.fem.inner_product_weights(
        mesh, quadrature_order=1).flatten(order="F")
    QL = sp.sparse.diags_array([qgL], offsets=[0])
    QL = sp.sparse.kron(sp.sparse.eye_array(3), QL)
    D = -G.T @ QL
    L = D @ G
    # Setup 1-step heat diffusion
    h = igl.avg_edge_length(V, C)
    dt = h**2
    k = 2
    A = M - k*dt*L
    # Precompute linear solvers
    Ainv = pbat.math.linalg.ldlt(A)
    Ainv.compute(A)
    Linv = pbat.math.linalg.ldlt(L)
    Linv.compute(L)
    # Setup isoline visuals
    niso = 10

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()

    def callback():
        global k, dt, Ainv, Linv, G, M, L, gamma, niso
        kchanged, k = imgui.InputFloat("k", k)
        if kchanged:
            A = M - k*dt*L
            Ainv.factorize(A)

        _, niso = imgui.InputInt("# iso", niso)
        _, gamma[0] = imgui.InputInt("source", gamma[0])
        if imgui.Button("Compute"):
            # Compute heat and its gradient
            u0 = np.zeros(n)
            u0[gamma] = 1
            b = M @ u0
            u = Ainv.solve(b).squeeze()
            gradu = (G @ u).reshape(int(G.shape[0]/3), 3)
            # Stable normalize gradient
            gradnorm = sp.linalg.norm(gradu, axis=1, keepdims=True)
            gnnz = gradnorm[:, 0] > 0
            gradu[gnnz, :] = gradu[gnnz, :] / gradnorm[gnnz, :]
            # Solve Poisson problem to reconstruct geodesic distance field, knowing that phi[0] = 0
            divGu = D @ gradu.reshape(G.shape[0])
            phi = Linv.solve(divGu).squeeze()
            # Laplacian is invariant to scale+translation, i.e. L(kx+t) = L(x).
            # This means that our solution can be shifted and/or reflected.
            # We handle this by flipping signs if a reflexion is "detected",
            # and shifting such that the smallest "distance" is 0.
            if phi[gamma].mean() > phi.mean():
                phi = -phi
            phi -= phi.min()

            # Code for libigl 2.5.1
            diso = (phi.max() - phi.min()) / niso
            isovalues = np.array([(i+0.5)*diso for i in range(niso)])
            Viso, Eiso, Iiso = igl.isolines(V, F, phi, isovalues)
            # Uncomment for libigl 2.4.1
            # Viso, Eiso = igl.isolines(V, F, phi, niso)
            cn = ps.register_curve_network("distance contours", Viso, Eiso)
            cn.set_color((0, 0, 0))
            cn.set_radius(0.002)
            vm = ps.get_volume_mesh(
                "model") if mesh.element == pbat.fem.Element.Tetrahedron else ps.get_surface_mesh("model")
            vm.add_scalar_quantity("heat", u, cmap="reds")
            vm.add_scalar_quantity("distance", phi, cmap="reds", enabled=True)
            grad_defined_on = "cells" if mesh.element == pbat.fem.Element.Tetrahedron else "faces"
            vm.add_vector_quantity("normalized heat grad",
                                   gradu, defined_on=grad_defined_on)
            vm.add_scalar_quantity("div unit gradient", divGu)

    if mesh.element == pbat.fem.Element.Tetrahedron:
        ps.register_volume_mesh("model", V, C)
    if mesh.element == pbat.fem.Element.Triangle:
        ps.register_surface_mesh("model", V, C)
    ps.set_user_callback(callback)
    ps.show()
