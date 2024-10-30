import pbatoolkit as pbat
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import igl
import time
import meshio
import argparse
import math


def min_max_eigs(A, n=10):
    lmax, Vmax = sp.sparse.linalg.eigsh(A, k=n, which='LA')
    lmin, Vmin = sp.sparse.linalg.eigsh(A, k=n, which='SA')
    return lmin, lmax


def shape_function_matrix(mesh, Xg):
    nevalpts = Xg.shape[1]
    # Unfortunately, we have to perform the copy here...
    # mesh.X and mesh.E are defined via def_property in pybind11, and
    # the underlying function returns a const reference. pbat.geometry.bvh
    # does not own any data, so we must make sure the V, C exist in memory.
    V = mesh.X
    C = mesh.E
    bvh = pbat.geometry.bvh(V, C, cell=pbat.geometry.Cell.Tetrahedron)
    e, d = bvh.nearest_primitives_to_points(Xg, parallelize=True)
    Xi = pbat.fem.reference_positions(mesh, e, Xg)
    phi = pbat.fem.shape_functions_at(mesh, Xi)
    # quad. pts. Xg that are outside the mesh should have N(Xg)=0
    # phi[:, np.array(d) > 0] = 0
    data = phi.flatten(order='F')
    rows = np.repeat(list(range(nevalpts)), mesh.E.shape[0])
    cols = mesh.E[:, e].flatten(order='F')
    nnodes = mesh.X.shape[1]
    N = sp.sparse.coo_matrix((data, (rows, cols)), shape=(nevalpts, nnodes))
    return N.asformat('csc')


class BaseFemFunctionTransferOperator():
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh):
        """Operator for transferring FEM discretized functions from a source 
        mesh MS to a target mesh MT, given the domain MD.

        Args:
            MD (pbat.fem.Mesh): Domain mesh
            MS (pbat.fem.Mesh): Source mesh
            MT (pbat.fem.Mesh): Target mesh
        """
        nelems = MT.E.shape[1]
        quadrature_order = 2*max(MS.order, MT.order)
        Xg = MT.quadrature_points(quadrature_order)
        wg = np.tile(MT.quadrature_weights(quadrature_order), nelems)
        Ig = sp.sparse.diags(wg)
        NS = shape_function_matrix(MS, Xg)
        NT = shape_function_matrix(MT, Xg)
        A = (NT.T @ Ig @ NT).asformat('csc')
        P = NT.T @ Ig @ NS
        self.Ig = Ig
        self.NS = NS
        self.NT = NT
        self.A = A
        self.P = P

    def __matmul__(self, X):
        pass


class CholFemFunctionTransferOperator(BaseFemFunctionTransferOperator):
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh):
        super().__init__(MD, MS, MT)
        n = self.A.shape[0]
        lmin, lmax = min_max_eigs(self.A, n=1)
        tau = 0.
        if lmin[0] <= 0:
            # Regularize A (due to positive semi-definiteness)
            tau = abs(lmin[0]) + 1e-10
        tau = sp.sparse.diags(np.full(n, tau))
        AR = self.A + tau
        solver = pbat.math.linalg.SolverBackend.Eigen
        self.Ainv = pbat.math.linalg.chol(AR, solver=solver)
        self.Ainv.compute(AR)

    def __matmul__(self, B):
        B = self.P @ B
        X = self.Ainv.solve(B)
        return X


class RankKApproximateFemFunctionTransferOperator(BaseFemFunctionTransferOperator):
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh, modes=30):
        super().__init__(MD, MS, MT)
        self.U, self.sigma, self.VT = sp.sparse.linalg.svds(
            self.Ig @ self.NT, k=modes, which='SM')
        keep = np.nonzero(self.sigma > 1e-5)[0]
        self.U = self.U[:, keep]
        self.sigma = self.sigma[keep]
        self.VT = self.VT[keep, :]

    def __matmul__(self, B):
        B = self.Ig @ self.NS @ B
        B = self.U.T @ B
        B = B / self.sigma[:, np.newaxis]
        X = self.VT.T @ B
        return X


def linear_elastic_deformation_modes(mesh, rho, Y, nu, modes=30, sigma=-1e-5):
    x = mesh.X.reshape(math.prod(mesh.X.shape), order='f')
    M, detJeM = pbat.fem.mass_matrix(mesh, rho=rho)
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep, detJeU, GNeU = pbat.fem.hyper_elastic_potential(
        mesh, Y, nu, energy=energy)
    hep.compute_element_elasticity(x)
    HU = hep.hessian()
    leigs, Veigs = sp.sparse.linalg.eigsh(
        HU, k=modes, M=M, sigma=sigma, which='LM')
    Veigs = Veigs / sp.linalg.norm(Veigs, axis=0, keepdims=True)
    leigs[leigs <= 0] = 0
    w = np.sqrt(leigs)
    return w, Veigs


def signal(w: float, v: np.ndarray, t: float, c: float, k: float):
    u = c*np.sin(k*w*t)*v
    return u


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Linear FEM shape transfer",
    )
    parser.add_argument("-i", "--input", help="Path to input tetrahedral mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-c", "--cage", help="Path to cage tetrahedral mesh", type=str,
                        dest="cage", required=True)
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    parser.add_argument("-k", "--num-modes", help="Number of modes to compute", type=int,
                        dest="modes", default=30)
    args = parser.parse_args()

    # Load input meshes
    imesh, icmesh = meshio.read(args.input), meshio.read(args.cage)
    V, C = imesh.points.astype(
        np.float64, order='c'), imesh.cells_dict["tetra"].astype(np.int64, order='c')
    CV, CC = icmesh.points.astype(
        np.float64, order='c'), icmesh.cells_dict["tetra"].astype(np.int64, order='c')
    maxcoord = V.max()
    V = V / maxcoord
    CV = CV / maxcoord
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
    CF = igl.boundary_facets(CC)
    CF[:, :2] = np.roll(CF[:, :2], shift=1, axis=1)
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron)
    cmesh = pbat.fem.Mesh(
        CV.T, CC.T, element=pbat.fem.Element.Tetrahedron)

    # Precompute quantities
    w, L = linear_elastic_deformation_modes(
        mesh, args.rho, args.Y, args.nu, args.modes)
    Fldl = CholFemFunctionTransferOperator(mesh, mesh, cmesh)
    Krestrict = 30
    Frank = RankKApproximateFemFunctionTransferOperator(
        mesh, mesh, cmesh, modes=Krestrict)

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_surface_mesh("model", V, F)
    ldlvm = ps.register_surface_mesh("LDL cage", CV, CF)
    rankvm = ps.register_surface_mesh("Rank K cage", CV, CF)
    mode = 6
    t0 = time.time()
    t = 0
    c = 0.15
    k = 0.05

    def callback():
        global mode, c, k
        changed, mode = imgui.InputInt("Mode", mode)
        changed, c = imgui.InputFloat("Wave amplitude", c)
        changed, k = imgui.InputFloat("Wave frequency", k)

        t = time.time() - t0
        X = V + signal(w[mode], L[:, mode],
                       t, c, k).reshape(V.shape[0], 3)
        XCldl = CV + Fldl @ (X - V)
        XCrank = CV + Frank @ (X - V)
        vm.update_vertex_positions(X)
        ldlvm.update_vertex_positions(XCldl)
        rankvm.update_vertex_positions(XCrank)

    ps.set_user_callback(callback)
    ps.show()
