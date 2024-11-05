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


def rest_pose_hessian(mesh, Y, nu):
    x = mesh.X.reshape(math.prod(mesh.X.shape), order='f')
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep, detJeU, GNeU = pbat.fem.hyper_elastic_potential(
        mesh, Y, nu, energy=energy)
    hep.compute_element_elasticity(x)
    HU = hep.hessian()
    return HU


def laplacian_energy(mesh, k=1, dims=1):
    L = -pbat.fem.laplacian(mesh, dims=dims)[0]
    M = None if k == 1 else pbat.fem.mass_matrix(mesh, dims=1, lump=True)
    U = L
    for i in range(k-1):
        U = U @ M @ L
    return U


def display_quadrature(mesh, Xg):
    V = mesh.X
    C = mesh.E
    bvh = pbat.geometry.bvh(V, C, cell=pbat.geometry.Cell.Tetrahedron)
    e, d = bvh.nearest_primitives_to_points(Xg, parallelize=True)
    nelems = mesh.E.shape[1]
    efree = np.setdiff1d(list(range(nelems)), np.unique(e))
    if len(efree) > 0:
        ps.register_volume_mesh(f"Invalid mesh nelems={
                                nelems}", V.T, C[:, efree].T)


def gradient_operator(mesh, Xg, dims=1):
    nevalpts = Xg.shape[1]
    nnodes = mesh.X.shape[1]
    V = mesh.X
    C = mesh.E
    bvh = pbat.geometry.bvh(V, C, cell=pbat.geometry.Cell.Tetrahedron)
    e, d = bvh.nearest_primitives_to_points(Xg, parallelize=True)
    Xi = pbat.fem.reference_positions(mesh, e, Xg)
    dphi = pbat.fem.shape_function_gradients_at(mesh, e, Xi)
    data = np.hstack([dphi[:, d::mesh.dims].flatten(order="F")
                     for d in range(mesh.dims)])
    rows = np.hstack([np.repeat(list(range(nevalpts)), dphi.shape[0]) + d*nevalpts
                     for d in range(mesh.dims)])
    cols = np.hstack([mesh.E[:, e].flatten(order="F")
                     for d in range(mesh.dims)])
    G = sp.sparse.coo_matrix((data, (rows, cols)),
                             shape=(mesh.dims*nevalpts, nnodes))
    if dims > 1:
        G = sp.sparse.kron(G, sp.sparse.eye(dims))
    return G.asformat('csc')


def shape_function_matrix(mesh, Xg, dims=1):
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
    if dims > 1:
        N = sp.sparse.kron(N, sp.sparse.eye(dims))
    return N.asformat('csc')


class BaseFemFunctionTransferOperator():
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh, H):
        """Operator for transferring FEM discretized functions from a source 
        mesh MS to a target mesh MT, given the domain MD.

        Args:
            MD (pbat.fem.Mesh): Domain mesh
            MS (pbat.fem.Mesh): Source mesh
            MT (pbat.fem.Mesh): Target mesh
        """
        quadrature_order = 2*max(MS.order, MT.order)
        Xg = MD.quadrature_points(quadrature_order)
        wg = np.tile(MD.quadrature_weights(quadrature_order), MD.E.shape[1])
        from scipy.sparse import kron, eye, diags
        Ig = diags(wg)
        Ig = kron(Ig, eye(MT.dims))
        NS = shape_function_matrix(MS, Xg, dims=MT.dims)
        NT = shape_function_matrix(MT, Xg, dims=MT.dims)
        A = (NT.T @ Ig @ NT).asformat('csc')
        P = NT.T @ Ig @ NS
        self.Ig = Ig
        self.NS = NS
        self.NT = NT
        self.A = A
        self.P = P
        self.U = laplacian_energy(MT, dims=MT.dims)
        self.GS = gradient_operator(MS, Xg, dims=MT.dims)
        self.GT = gradient_operator(MT, Xg, dims=MT.dims)
        self.IG = kron(kron(eye(MT.dims), diags(wg)), eye(MT.dims))
        self.H = H

    def __matmul__(self, X):
        pass


class CholFemFunctionTransferOperator(BaseFemFunctionTransferOperator):
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh, H, lreg=5, hreg=1, greg=1, Y=1e6, nu=0.45, hxreg=1e-4, rho=1e3):
        super().__init__(MD, MS, MT, H)
        n = self.A.shape[0]
        energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
        self.hep, self.detJeU, self.GNeU = pbat.fem.hyper_elastic_potential(
            MT, Y, nu, energy=energy)
        self.MT = MT
        self.rho = rho
        self.M = self.NT.T @ (rho * self.Ig) @ self.NT  # + lreg*self.U + hreg*self.H + greg * self.GT.T @ self.IG @ self.GT
        self.P = self.NT.T @ (rho * self.Ig) @ self.NS
        self.hxreg = hxreg
        # lmin, lmax = min_max_eigs(A, n=1)
        # tau = 0.
        # if lmin[0] <= 0:
        #     # Regularize A (due to positive semi-definiteness)
        #     print("WARNING: A is semi-definite")
        #     tau = abs(lmin[0]) + 1e-10
        # tau = sp.sparse.diags(np.full(n, tau))
        # AR = A + tau
        # solver = pbat.math.linalg.SolverBackend.Eigen
        # self.Ainv = pbat.math.linalg.chol(AR, solver=solver)
        # self.Ainv.compute(AR)
        self.uk = np.zeros(math.prod(self.MT.X.shape))

    def __matmul__(self, u):
        self.uk = np.zeros(math.prod(self.MT.X.shape))
        xr = self.MT.X.reshape(math.prod(self.MT.X.shape), order='f')
        for k in range(1):
            x = xr + self.uk
            self.hep.compute_element_elasticity(x)
            duku = self.NT @ self.uk - self.NS @ u
            E = 0.5 * duku.T @ (self.rho * self.Ig) @ duku + self.hxreg * self.hep.eval()
            H = self.M + self.hxreg * self.hep.hessian()
            solver = pbat.math.linalg.SolverBackend.Eigen
            Hinv = pbat.math.linalg.ldlt(H, solver=solver)
            Hinv.compute(H)
            # B = self.P @ u + self.GT.T @ self.IG @ self.GS @ u
            ng = self.P @ u - self.hxreg * self.hep.gradient()
            r = ng.T @ ng
            if r < 1e-3:
                break
            # X = self.Ainv.solve(B)
            du = Hinv.solve(ng).squeeze()
            self.uk += du
        return self.uk


class RankKApproximateFemFunctionTransferOperator(BaseFemFunctionTransferOperator):
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh, H, lreg=5, hreg=1, greg=1, modes=30):
        super().__init__(MD, MS, MT, H)
        A = self.A + lreg*self.U + hreg*self.H + greg * self.GT.T @ self.IG @ self.GT
        l, V = sp.sparse.linalg.eigsh(A, k=modes, sigma=1e-5, which='LM')
        keep = np.nonzero(l > 1e-5)[0]
        self.l, self.V = l[keep], V[:, keep]

    def __matmul__(self, B):
        B = self.P @ B + self.GT.T @ self.IG @ self.GS @ B
        B = self.V.T @ B
        B = B @ sp.sparse.diags(1 / self.l)
        X = self.V @ B
        return X


def linear_elastic_deformation_modes(mesh, rho, Y, nu, modes=30, sigma=-1e-5):
    M, detJeM = pbat.fem.mass_matrix(mesh, rho=rho)
    HU = rest_pose_hessian(mesh, Y, nu)
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
    HC = rest_pose_hessian(cmesh, args.Y, args.nu)
    lreg, hreg, greg, hxreg = 0, 0, 0, 1e-4
    Fldl = CholFemFunctionTransferOperator(
        mesh, mesh, cmesh, HC, lreg=lreg, hreg=hreg, greg=greg, hxreg=hxreg, rho=args.rho)
    # Krestrict = 30
    # Frank = RankKApproximateFemFunctionTransferOperator(
    #     mesh, mesh, cmesh, HC, lreg=lreg, hreg=hreg, greg=greg, modes=Krestrict)

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_surface_mesh("model", V, F)
    ldlvm = ps.register_surface_mesh("LDL cage", CV, CF)
    # rankvm = ps.register_surface_mesh("Rank K cage", CV, CF)
    mode = 6
    t0 = time.time()
    t = 0
    c = 0.15
    k = 0.05
    theta = 0
    dtheta = np.pi/120
    animate = False
    step = False

    def callback():
        global mode, c, k, theta, dtheta
        global animate, step
        
        changed, mode = imgui.InputInt("Mode", mode)
        changed, c = imgui.InputFloat("Wave amplitude", c)
        changed, k = imgui.InputFloat("Wave frequency", k)
        changed, animate = imgui.Checkbox("animate", animate)
        step = imgui.Button("step")

        if animate or step:
            t = time.time() - t0

            R = sp.spatial.transform.Rotation.from_quat(
                [0, np.sin(theta/2), 0, np.cos(theta/4)]).as_matrix()
            X = (V - V.mean(axis=0)) @ R.T + V.mean(axis=0)
            uf = signal(w[mode], L[:, mode], t, c, k)
            ur = (X - V).flatten(order="C")
            ut = 1
            u = uf + ur + ut
            XCldl = CV + (Fldl @ u).reshape(CV.shape)
            # XCrank = CV + (Frank @ u).reshape(CV.shape)
            vm.update_vertex_positions(V + (uf + ur).reshape(V.shape))
            ldlvm.update_vertex_positions(XCldl)
            # rankvm.update_vertex_positions(XCrank)

            theta += dtheta
            if theta > 2*np.pi:
                theta = 0

    ps.set_user_callback(callback)
    ps.show()
