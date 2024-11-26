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
import qpsolvers


def min_max_eigs(A, n=10):
    lmax, Vmax = sp.sparse.linalg.eigsh(A, k=n, which='LA')
    lmin, Vmin = sp.sparse.linalg.eigsh(A, k=n, which='SA')
    return lmin, lmax


def laplacian_energy(mesh, k=1, dims=1):
    L = -pbat.fem.laplacian(mesh, dims=dims)[0]
    M = None if k == 1 else pbat.fem.mass_matrix(mesh, dims=1, lump=True)
    U = L
    for i in range(k-1):
        U = U @ M @ L
    return U


def transfer_quadrature(cmesh, wg2, Xg2, order=1):
    CV, CC = cmesh.X, cmesh.E
    Xg1 = cmesh.quadrature_points(order)
    e1 = np.linspace(0, cmesh.E.shape[1]-1,
                     num=cmesh.E.shape[1], dtype=np.int64)
    e1 = np.repeat(e1, int(Xg1.shape[1] / cmesh.E.shape[1]))
    Xi1 = pbat.fem.reference_positions(cmesh, e1, Xg1)
    bvh1 = pbat.geometry.bvh(CV, CC, cell=pbat.geometry.Cell.Tetrahedron)
    e2 = bvh1.nearest_primitives_to_points(Xg2, parallelize=True)[0]
    Xi2 = pbat.fem.reference_positions(cmesh, e2, Xg2)
    wg1, err = pbat.math.transfer_quadrature(
        e1, Xi1, e2, Xi2, wg2, order=order, with_error=True, max_iters=50, precision=1e-10)
    return wg1, Xg1, e1, err


def patch_quadrature(cmesh, wg1, Xg1, e1, err=1e-4, numerical_zero=1e-10):
    wtotal = np.zeros(cmesh.E.shape[1])
    np.add.at(wtotal, e1, wg1)
    ezero = np.nonzero(wtotal < numerical_zero)
    ezeromask = np.in1d(e1, ezero)
    e1zero = np.copy(e1[ezeromask])
    Xg1zero = np.copy(Xg1[:, ezeromask])
    MM, BM, PM = pbat.math.reference_moment_fitting_systems(
        e1zero, Xg1zero, e1zero, Xg1zero, np.zeros(Xg1zero.shape[1]))
    MM = pbat.math.block_diagonalize_moment_fitting(MM, PM)
    n = np.count_nonzero(ezeromask)
    P = -sp.sparse.eye(n).asformat('csc')
    G = MM.asformat('csc')
    h = np.full(G.shape[0], err)
    lb = np.full(n, 0.)
    q = np.full(n, 0.)
    wzero = qpsolvers.solve_qp(P, q, G=G, h=h, lb=lb, initvals=np.zeros(
        n), solver="cvxopt")
    wg1p = np.copy(wg1)
    wg1p[ezeromask] = wzero
    return wg1p, ezeromask


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


def shape_functions(mesh, Xg, dims=1):
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
    phi[:, d > 0] = 0
    return phi, e


def shape_function_matrix(mesh, Xg, dims=1):
    nevalpts = Xg.shape[1]
    phi, e = shape_functions(mesh, Xg, dims=dims)
    data = phi.flatten(order='F')
    rows = np.repeat(list(range(nevalpts)), mesh.E.shape[0])
    cols = mesh.E[:, e].flatten(order='F')
    nnodes = mesh.X.shape[1]
    N = sp.sparse.coo_matrix((data, (rows, cols)), shape=(nevalpts, nnodes))
    if dims > 1:
        N = sp.sparse.kron(N, sp.sparse.eye(dims))
    return N.asformat('csc')


def shape_function_gradients(mesh, Xg):
    V = mesh.X
    C = mesh.E
    bvh = pbat.geometry.bvh(V, C, cell=pbat.geometry.Cell.Tetrahedron)
    e, d = bvh.nearest_primitives_to_points(Xg, parallelize=True)
    Xi = pbat.fem.reference_positions(mesh, e, Xg)
    gradphi = pbat.fem.shape_function_gradients_at(mesh, e, Xi)
    return gradphi, e


class BaseFemFunctionTransferOperator():
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh, H=None):
        """Operator for transferring FEM discretized functions from a source 
        mesh MS to a target mesh MT, given the domain MD.

        Args:
            MD (pbat.fem.Mesh): Domain mesh
            MS (pbat.fem.Mesh): Source mesh
            MT (pbat.fem.Mesh): Target mesh
        """
        quadrature_order = 2*max(MS.order, MT.order)
        Xg = MD.quadrature_points(quadrature_order)
        wg = pbat.fem.inner_product_weights(
            MD, quadrature_order=quadrature_order
        ).flatten(order="F")
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


class NonLinearElasticRestrictionEnergy:
    def __init__(self, hep, rho, Ig, NT, y, xr, hxreg):
        self.hep = hep
        self.rho = rho
        self.Ig = Ig
        self.NT = NT
        self.y = y
        self.xr = xr
        self.hxreg = hxreg

    def __call__(self, u):
        x = self.xr + u
        self.hep.compute_element_elasticity(x, grad=False, hessian=False)
        duku = self.NT @ u - self.y
        E = 0.5 * duku.T @ (self.rho * self.Ig) @ duku + \
            self.hxreg * self.hep.eval()
        return E


class CholFemFunctionTransferOperator(BaseFemFunctionTransferOperator):
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh, H, lreg=5, hreg=1, greg=1):
        super().__init__(MD, MS, MT, H)
        n = self.A.shape[0]
        self.greg = greg
        A = self.A + lreg*self.U + hreg*self.H + greg * self.GT.T @ self.IG @ self.GT
        lmin, lmax = min_max_eigs(A, n=1)
        tau = 0.
        if lmin[0] <= 0:
            # Regularize A (due to positive semi-definiteness)
            tau = abs(lmin[0]) + 1e-10
        tau = sp.sparse.diags(np.full(n, tau))
        AR = A + tau
        solver = pbat.math.linalg.SolverBackend.Eigen
        self.Ainv = pbat.math.linalg.chol(AR, solver=solver)
        self.Ainv.compute(AR)

    def __matmul__(self, u):
        b = self.P @ u + self.greg * self.GT.T @ self.IG @ self.GS @ u
        du = self.Ainv.solve(b).squeeze()
        return du


class RankKApproximateFemFunctionTransferOperator(BaseFemFunctionTransferOperator):
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh, H, lreg=5, hreg=1, greg=1, modes=30):
        super().__init__(MD, MS, MT, H)
        self.greg = greg
        A = self.A + lreg*self.U + hreg*self.H + greg * self.GT.T @ self.IG @ self.GT
        l, V = sp.sparse.linalg.eigsh(A, k=modes, sigma=1e-5, which='LM')
        keep = np.nonzero(l > 1e-5)[0]
        self.l, self.V = l[keep], V[:, keep]

    def __matmul__(self, B):
        B = self.P @ B + self.greg * self.GT.T @ self.IG @ self.GS @ B
        B = self.V.T @ B
        B = B @ sp.sparse.diags(1 / self.l)
        X = self.V @ B
        return X


class NewtonFunctionTransferOperator(BaseFemFunctionTransferOperator):
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh, hxreg=1e-4, Y=1e6, nu=0.45, rho=1e3):
        super().__init__(MD, MS, MT)
        self.M = (self.NT.T @ (rho * self.Ig) @ self.NT).asformat('csc')
        self.P = (self.NT.T @ (rho * self.Ig) @ self.NS).asformat('csc')
        self.rho = rho
        energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
        self.hep, self.egU, self.wgU, self.GNeU = pbat.fem.hyper_elastic_potential(
            MT, Y, nu, energy=energy)
        self.hxreg = hxreg
        self.X = MT.X
        self.u = np.zeros(math.prod(self.X.shape), order="f")

    def __matmul__(self, u):
        xr = self.X.reshape(math.prod(self.X.shape), order='f')
        uk = self.u  # np.zeros_like(xr)
        y = self.P @ u
        f = NonLinearElasticRestrictionEnergy(
            self.hep, self.rho, self.Ig, self.NT, self.NS @ u, xr, self.hxreg)
        for k in range(5):
            x = xr + uk
            self.hep.compute_element_elasticity(x)
            K = self.hep.hessian()
            H = self.M + self.hxreg * K
            solver = pbat.math.linalg.SolverBackend.Eigen
            Hinv = pbat.math.linalg.ldlt(H, solver=solver)
            Hinv.compute(H)
            gk = (self.M @ uk - y) + self.hxreg * self.hep.gradient()
            du = Hinv.solve(-gk).squeeze()
            # Line search
            alpha = 1.
            Dfk = gk.dot(du)
            fk = f(uk)
            maxiters = 20
            tau = 0.5
            for j in range(maxiters):
                fx = f(uk + alpha*du)
                flinear = fk + alpha * c * Dfk
                if fx <= flinear:
                    break
                alpha = tau*alpha
            uk += alpha*du
        self.u = uk
        return uk


class VbdFunctionTransferOperator:
    def __init__(self, MD: pbat.fem.Mesh, MS: pbat.fem.Mesh, MT: pbat.fem.Mesh, hxreg=1e-4, Y=1e6, nu=0.45, rho=1e3):
        quadrature_order = 1
        wgD = pbat.fem.inner_product_weights(
            MD, quadrature_order).flatten(order="F")
        XgD = MD.quadrature_points(quadrature_order)
        wgT, self.Xg, self.eg, errT = transfer_quadrature(
            MT, wgD, XgD, order=quadrature_order)
        self.wg, ezeroinds = patch_quadrature(
            MT, wgT, self.Xg, self.eg, err=1e-4)
        self.Ng, eNg = shape_functions(MT, self.Xg)
        self.GNeg, eGNeg = shape_function_gradients(MT, self.Xg)
        Y = np.full(self.wg.shape[0], Y)
        nu = np.full(self.wg.shape[0], nu)
        self.mug = Y / (2*(1+nu))
        self.lambdag = (Y*nu) / ((1+nu)*(1-2*nu))
        self.NT = shape_function_matrix(MT, self.Xg)
        self.NS = shape_function_matrix(MS, self.Xg)
        Ig = sp.sparse.diags_array(rho * self.wg)
        M = (self.NT.T @ Ig @ self.NT).asformat('csc')
        self.m = np.array(M.sum(axis=0)).squeeze()
        self.P = (self.NT.T @ Ig @ self.NS).asformat('csc')
        self.rho = rho
        energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
        self.hep, self.egU, self.wgU, self.GNeU = pbat.fem.hyper_elastic_potential(
            MT, Y, nu, energy=energy, eg=self.eg, wg=self.wg, GNeg=self.GNeg)
        GNegS, egS = shape_function_gradients(MS, self.Xg)
        self.hepS, self.egUS, self.wgUS, self.GNeUS = pbat.fem.hyper_elastic_potential(
            MS, Y, nu, energy=energy, eg=egS, wg=self.wg, GNeg=GNegS)
        self.Kpsi = hxreg
        self.R = pbat.sim.vbd.Restriction()
        self.R.E = MT.E
        self.R.eg = self.eg
        self.R.Ng = self.Ng
        self.R.GNeg = self.GNeg
        self.R.wg = self.wg
        self.R.rhog = np.full(self.wg.shape[0], rho)
        self.R.mug = self.mug
        self.R.lambdag = self.lambdag
        self.R.m = self.m
        self.R.Kpsi = self.Kpsi
        V, C = MT.X.T, MT.E[:, self.eg].T
        eg = np.repeat(self.eg, C.shape[1])
        ilocal = np.tile(np.arange(C.shape[1]), self.eg.shape[0])
        GVTe = pbat.sim.vbd.vertex_element_adjacency(V, C, data=eg)
        GVTi = pbat.sim.vbd.vertex_element_adjacency(V, C, data=ilocal)
        self.R.GVGp = GVTi.indptr
        self.R.GVGg = GVTi.indices
        self.R.GVGe = GVTe.data
        self.R.GVGilocal = GVTi.data
        self.R.partitions = pbat.sim.vbd.partitions(V, C)[0]
        self.X = MT.X
        self.XS = MS.X
        self.u = np.zeros(math.prod(self.X.shape), order="f")

    def __matmul__(self, u):
        xrs = self.XS.reshape(math.prod(self.XS.shape), order='f')
        uk = self.u
        U = u.reshape((3, int(u.shape[0] / 3)), order="F")
        Uk = uk.reshape(self.X.shape, order="F")
        xtildeg = U @ self.NS.T
        self.R.set_target_shape(xtildeg)
        xs = xrs + u
        self.hepS.compute_element_elasticity(xs, grad=False, hessian=False)
        self.R.Psitildeg = self.hepS.Ug
        Ukp1 = self.R.apply(Uk, iters=20)
        self.uk = Ukp1.flatten(order="F")
        return self.uk


def rest_pose_hessian(mesh, Y, nu):
    x = mesh.X.reshape(math.prod(mesh.X.shape), order='f')
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep, egU, wgU, GNeU = pbat.fem.hyper_elastic_potential(
        mesh, Y, nu, energy=energy)
    hep.compute_element_elasticity(x)
    HU = hep.hessian()
    return HU


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
    lreg, hreg, greg, hxreg = 1e-2, 0, 1, 1e-4
    # Fnewton = CholFemFunctionTransferOperator(
    #     mesh, mesh, cmesh, HC, lreg=lreg, hreg=hreg, greg=greg)
    # Fnewton = NewtonFunctionTransferOperator(
    #     mesh, mesh, cmesh, hxreg=hxreg)
    # Krestrict = 30
    # Frank = RankKApproximateFemFunctionTransferOperator(
    #     mesh, mesh, cmesh, HC, lreg=lreg, hreg=hreg, greg=greg, modes=Krestrict)
    Fvbd = VbdFunctionTransferOperator(mesh, mesh, cmesh, hxreg=1e-8)

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_surface_mesh("model", V, F)
    # newtonvm = ps.register_surface_mesh("Newton cage", CV, CF)
    # newtonvm.set_transparency(0.5)
    # newtonvm.set_edge_width(1)
    # rankvm = ps.register_surface_mesh("Rank K cage", CV, CF)
    # rankvm.set_transparency(0.5)
    # rankvm.set_edge_width(1)
    vbdvm = ps.register_surface_mesh("VBD cage", CV, CF)
    vbdvm.set_transparency(0.5)
    vbdvm.set_edge_width(1)
    mode = 6
    t0 = time.time()
    t = 0
    c = 0.15
    k = 0.05
    theta = 0
    dtheta = np.pi/120
    animate = False
    step = False
    screenshot = False

    # wg2 = pbat.fem.inner_product_weights(mesh, 1).flatten(order="F")
    # Xg2 = mesh.quadrature_points(1)

    # pg2 = ps.register_point_cloud("Fine quad", Xg2.T)
    # pg2.add_scalar_quantity("weights", wg2, enabled=True)
    # pg2.set_point_radius_quantity("weights")

    # wg1, Xg1, e1, err = transfer_quadrature(cmesh, wg2, Xg2, order=1)
    # pg1 = ps.register_point_cloud("Cage quad", Xg1.T)
    # pg1.add_scalar_quantity("weights", wg1, enabled=True)
    # pg1.set_point_radius_quantity("weights")

    # wg1p, ezeroinds = patch_quadrature(cmesh, wg1, Xg1, e1, err=1e-4)
    # ezero = np.unique(e1[ezeroinds])
    # wgpatched = wg1p[ezeroinds]
    # Xgpatched = Xg1[:, ezeroinds]
    # pp = ps.register_point_cloud("Patched", Xgpatched.T)
    # pp.add_scalar_quantity("weights", wgpatched, enabled=True)
    # pp.set_point_radius_quantity("weights")
    # ps.register_volume_mesh("Bad coarse", CV, CC[ezero, :])
    # pg1p = ps.register_point_cloud("Cage quad patched", Xg1.T)
    # pg1p.add_scalar_quantity("weights", wg1p, enabled=True)
    # pg1p.set_point_radius_quantity("weights")

    def callback():
        global mode, c, k, theta, dtheta
        global animate, step, screenshot
        changed, mode = imgui.InputInt("Mode", mode)
        changed, c = imgui.InputFloat("Wave amplitude", c)
        changed, k = imgui.InputFloat("Wave frequency", k)
        changed, animate = imgui.Checkbox("animate", animate)
        changed, screenshot = imgui.Checkbox("screenshot", screenshot)
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
            # XCnewton = CV + (Fnewton @ u).reshape(CV.shape)
            XCvbd = CV + (Fvbd @ u).reshape(CV.shape)
            # XCrank = CV + (Frank @ u).reshape(CV.shape)
            vm.update_vertex_positions(V + (uf + ur).reshape(V.shape))
            # newtonvm.update_vertex_positions(XCnewton)
            vbdvm.update_vertex_positions(XCvbd)
            # rankvm.update_vertex_positions(XCrank)

            theta += dtheta
            if theta > 2*np.pi:
                theta = 0

        if screenshot:
            ps.screenshot()

    ps.set_user_callback(callback)
    ps.show()
