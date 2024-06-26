import pbatoolkit as pbat
import pbatoolkit.fem
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import time
import meshio
import argparse


def signal(w: float, v: np.ndarray, t: float, c: float, k: float):
    u = c*np.sin(k*w*t)*v
    return u


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Elastic stiffness modal decomposition demo",
    )
    parser.add_argument("-i", "--input", help="Path to input tetrahedral mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    parser.add_argument("-k", "--num-modes", help="Number of modes to compute", type=int,
                        dest="modes", default=30)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    mesh = pbat.fem.mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    x = mesh.X.reshape(mesh.X.shape[0]*mesh.X.shape[1], order='f')
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    M = pbat.fem.mass_matrix(mesh, detJeM, rho=args.rho,
                             dims=3, quadrature_order=2).to_matrix()
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    hep = pbat.fem.hyper_elastic_potential(
        mesh, detJeU, GNeU, Y, nu, psi=pbat.fem.HyperElasticEnergy.StableNeoHookean, quadrature_order=1)
    hep.precompute_hessian_sparsity()
    hep.compute_element_elasticity(x)
    U, gradU, HU = hep.eval(), hep.to_vector(), hep.to_matrix()
    leigs, Veigs = sp.sparse.linalg.eigsh(HU, k=args.modes, M=M, which='SM')
    leigs[leigs <= 0] = 0
    w = np.sqrt(leigs)

    ps.init()
    vm = ps.register_volume_mesh("model", mesh.X.T, mesh.E.T)
    mode = 6
    t0 = time.time()
    t = 0
    c = 0.15
    k = 0.05

    def callback():
        global mode, c, k
        changed, mode = imgui.InputInt("mode", mode)
        changed, c = imgui.SliderFloat("c", c, v_min=0, v_max=10)
        changed, k = imgui.SliderFloat("k", k, v_min=0, v_max=1)

        t = time.time() - t0
        X = mesh.X.T + signal(w[mode], Veigs[:, mode],
                              t, c, k).reshape(mesh.X.shape[1], 3)
        vm.update_vertex_positions(X)

    ps.set_user_callback(callback)
    ps.show()
