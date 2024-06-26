import pbatoolkit as pbat
import pbatoolkit.fem
import pbatoolkit.math.linalg
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
    parser.add_argument("-i", "--input", help="Path to input tetrahedral mesh", type=str,
                        dest="input", required=True)
    args = parser.parse_args()
    
    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    gamma = [0]
    # Construct Galerkin laplacian, mass and gradient operators
    mesh = pbat.fem.mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    detJeL = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeL = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    L = pbat.fem.laplacian_matrix(
        mesh, detJeL, GNeL, quadrature_order=1).to_matrix()
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    M = pbat.fem.mass_matrix(mesh, detJeM, dims=1,
                             quadrature_order=2).to_matrix()
    G = pbat.fem.galerkin_gradient_matrix(
        mesh, detJeL, GNeL, quadrature_order=1).to_matrix()
    # Setup 1-step heat diffusion
    h = igl.avg_edge_length(mesh.X.T, mesh.E.T)
    dt = h**2
    k = 2
    A = M - k*dt*L
    # n = mesh.X.shape[1]
    # u0 = np.zeros(n)
    # u0[gamma] = 1
    # b = M @ u0
    # Precompute linear solvers
    Ainv = pbat.math.linalg.ldlt(A)
    Ainv.compute(A)
    Linv = pbat.math.linalg.ldlt(L)
    Linv.compute(L)
    # # Compute heat and its gradient
    # u = Ainv.solve(b).squeeze()
    # gradu = (G @ u).reshape(int(G.shape[0]/3), 3)
    # # Stable normalize gradient
    # gradnorm = sp.linalg.norm(gradu, axis=1, keepdims=True)
    # gradu = gradu / gradnorm
    # # Solve Poisson problem to reconstruct geodesic distance field, knowing that phi[0] = 0
    # divGu = -G.T @ gradu.reshape(G.shape[0])
    # phi = Linv.solve(divGu).squeeze()
    # # Laplacian is invariant to scale+translation, i.e. L(kx+t) = L(x).
    # # This means that our solution can be shifted and/or reflected.
    # # We handle this by flipping signs if a reflexion is "detected",
    # # and shifting such that the smallest "distance" is 0.
    # if phi[gamma].mean() > phi.mean():
    #     phi = -phi
    # phi -= phi.min()

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.init()
    
    def callback():
        global k, dt, Ainv, Linv, G, M, L, gamma
        _, k = imgui.SliderFloat("k", k, v_min=0, v_max=5)
        _, gamma[0] = imgui.InputInt("source", gamma[0])
        if imgui.Button("Compute"):
            # Compute heat and its gradient
            n = M.shape[0]
            u0 = np.zeros(n)
            u0[gamma] = 1
            b = M @ u0
            u = Ainv.solve(b).squeeze()
            gradu = (G @ u).reshape(int(G.shape[0]/3), 3)
            # Stable normalize gradient
            gradnorm = sp.linalg.norm(gradu, axis=1, keepdims=True)
            gradu = gradu / gradnorm
            # Solve Poisson problem to reconstruct geodesic distance field, knowing that phi[0] = 0
            divGu = -G.T @ gradu.reshape(G.shape[0])
            phi = Linv.solve(divGu).squeeze()
            # Laplacian is invariant to scale+translation, i.e. L(kx+t) = L(x).
            # This means that our solution can be shifted and/or reflected.
            # We handle this by flipping signs if a reflexion is "detected",
            # and shifting such that the smallest "distance" is 0.
            if phi[gamma].mean() > phi.mean():
                phi = -phi
            phi -= phi.min()
            
            vm = ps.get_volume_mesh("model")
            vm.add_scalar_quantity("heat", u, cmap="reds")
            vm.add_scalar_quantity("distance", phi, cmap="reds", enabled=True)
            vm.add_vector_quantity("normalized heat grad", gradu)
            vm.add_scalar_quantity("div unit gradient", divGu)
            
            
    vm = ps.register_volume_mesh("model", mesh.X.T, mesh.E.T)
    ps.set_user_callback(callback)
    ps.show()
