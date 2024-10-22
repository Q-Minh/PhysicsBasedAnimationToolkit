import pbatoolkit as pbat
import igl
import ipctk
import meshio
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import math
import argparse
import itertools
from collections.abc import Callable


def combine(V: list, C: list):
    Vsizes = [Vi.shape[0] for Vi in V]
    offsets = list(itertools.accumulate(Vsizes))
    C = [C[i] + offsets[i] - Vsizes[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    return V, C


def line_search(alpha0: float,
                xk: np.ndarray,
                dx: np.ndarray,
                gk: np.ndarray,
                f: Callable[[np.ndarray], float],
                maxiters: int = 20,
                c: float = 1e-4,
                tau: float = 0.5):
    alphaj = alpha0
    Dfk = gk.dot(dx)
    fk = f(xk)
    for j in range(maxiters):
        fx = f(xk + alphaj*dx)
        flinear = fk + alphaj * c * Dfk
        if fx <= flinear:
            break
        alphaj = tau*alphaj
    return alphaj


def newton(x0: np.ndarray,
           f: Callable[[np.ndarray], float],
           grad: Callable[[np.ndarray], np.ndarray],
           hess: Callable[[np.ndarray], sp.sparse.csc_matrix],
           lsolver: Callable[[sp.sparse.csc_matrix, np.ndarray], np.ndarray],
           alpha0: Callable[[np.ndarray, np.ndarray], float],
           maxiters: int = 10,
           rtol: float = 1e-5,
           callback: Callable[[np.ndarray], None] = None):
    xk = x0
    gk = grad(x0)
    for k in range(maxiters):
        gnorm = np.linalg.norm(gk, 1)
        if gnorm < rtol:
            break
        Hk = hess(xk)
        dx = lsolver(Hk, -gk)
        alpha = line_search(alpha0(xk, dx), xk, dx, gk, f)
        xk = xk + alpha*dx
        gk = grad(xk)
        if callback is not None:
            callback(xk)
    return xk


def to_surface(x: np.ndarray, mesh: pbat.fem.Mesh, cmesh: ipctk.CollisionMesh):
    X = x.reshape(mesh.X.shape[0],
                  mesh.X.shape[1], order='F').T
    XB = cmesh.map_displacements(X)
    return XB


class Parameters():
    def __init__(self,
                 mesh: pbat.fem.Mesh,
                 xt: np.ndarray,
                 vt: np.ndarray,
                 a: np.ndarray,
                 M: sp.sparse.dia_array,
                 hep: pbat.fem.HyperElasticPotential,
                 dt: float,
                 cmesh: ipctk.CollisionMesh,
                 cconstraints: ipctk.CollisionConstraints,
                 fconstraints: ipctk.FrictionConstraints,
                 dhat: float = 1e-3,
                 dmin: float = 1e-4,
                 mu: float = 0.3,
                 epsv: float = 1e-4):
        self.mesh = mesh
        self.xt = xt
        self.vt = vt
        self.a = a
        self.M = M
        self.hep = hep
        self.dt = dt
        self.cmesh = cmesh
        self.cconstraints = cconstraints
        self.fconstraints = fconstraints
        self.dhat = dhat
        self.dmin = dmin
        self.mu = mu
        self.epsv = epsv

        self.dt2 = dt**2
        self.xtilde = xt + dt*vt + self.dt2 * a
        self.avgmass = M.diagonal().mean()
        self.kB = None
        self.maxkB = None
        self.dprev = None
        self.dcurrent = None
        BX = to_surface(xt, mesh, cmesh)
        self.bboxdiag = ipctk.world_bbox_diagonal_length(BX)
        self.gU = None
        self.gB = None


class Potential():
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray) -> float:
        dt = self.params.dt
        dt2 = self.params.dt2
        xt = self.params.xt
        xtilde = self.params.xtilde
        M = self.params.M
        hep = self.params.hep
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        cconstraints = self.params.cconstraints
        fconstraints = self.params.fconstraints
        dhat = self.params.dhat
        dmin = self.params.dmin
        mu = self.params.mu
        epsv = self.params.epsv
        kB = self.params.kB

        hep.compute_element_elasticity(x, grad=False, hessian=False)
        U = hep.eval()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        BXdot = to_surface(v, mesh, cmesh)
        cconstraints.build(cmesh, BX, dhat, dmin=dmin)
        fconstraints.build(cmesh, BX, cconstraints, dhat, kB, mu)
        EB = cconstraints.compute_potential(cmesh, BX, dhat)
        EF = fconstraints.compute_potential(cmesh, BXdot, epsv)
        return 0.5 * (x - xtilde).T @ M @ (x - xtilde) + dt2*U + kB * EB + dt2*EF


class Gradient():
    def __init__(self, params: Parameters):
        self.params = params
        self.gradU = None
        self.gradB = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        dt = self.params.dt
        dt2 = self.params.dt2
        xt = self.params.xt
        xtilde = self.params.xtilde
        M = self.params.M
        hep = self.params.hep
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        cconstraints = self.params.cconstraints
        fconstraints = self.params.fconstraints
        dhat = self.params.dhat
        dmin = self.params.dmin
        mu = self.params.mu
        epsv = self.params.epsv
        kB = self.params.kB

        hep.compute_element_elasticity(x, grad=True, hessian=False)
        gU = hep.gradient()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        cconstraints.build(cmesh, BX, dhat, dmin=dmin)
        gB = cconstraints.compute_potential_gradient(cmesh, BX, dhat)
        gB = cmesh.to_full_dof(gB)

        # Cannot compute gradient without barrier stiffness
        if self.params.kB is None:
            binit = BarrierInitializer(self.params)
            binit(x, gU, gB)

        kB = self.params.kB
        BXdot = to_surface(v, mesh, cmesh)
        fconstraints.build(cmesh, BX, cconstraints, dhat, kB, mu)
        gF = fconstraints.compute_potential_gradient(cmesh, BXdot, epsv)
        gF = cmesh.to_full_dof(gF)
        g = M @ (x - xtilde) + dt2*gU + kB * gB + dt*gF
        return g


class Hessian():
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray) -> sp.sparse.csc_matrix:
        dt = self.params.dt
        dt2 = self.params.dt2
        xt = self.params.xt
        M = self.params.M
        hep = self.params.hep
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        cconstraints = self.params.cconstraints
        fconstraints = self.params.fconstraints
        dhat = self.params.dhat
        dmin = self.params.dmin
        mu = self.params.mu
        epsv = self.params.epsv
        kB = self.params.kB

        hep.compute_element_elasticity(x, grad=False, hessian=True)
        HU = hep.hessian()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        BXdot = to_surface(v, mesh, cmesh)
        HB = cconstraints.compute_potential_hessian(
            cmesh, BX, dhat, project_hessian_to_psd=True)
        HB = cmesh.to_full_dof(HB)
        HF = fconstraints.compute_potential_hessian(
            cmesh, BXdot, epsv, project_hessian_to_psd=True)
        HF = cmesh.to_full_dof(HF)
        H = M + dt2*HU + kB * HB + HF
        return H


class LinearSolver():

    def __init__(self, dofs: np.ndarray):
        self.dofs = dofs

    def __call__(self, A: sp.sparse.csc_matrix, b: np.ndarray) -> np.ndarray:
        dofs = self.dofs
        Add = A.tocsr()[dofs, :].tocsc()[:, dofs]
        bd = b[dofs]
        Addinv = pbat.math.linalg.ldlt(Add)
        Addinv.compute(Add)
        # NOTE: If built from source with SuiteSparse, use faster chol
        # Addinv = pbat.math.linalg.chol(
        #     Add, solver=pbat.math.linalg.SolverBackend.SuiteSparse)
        # Addinv.compute(sp.sparse.tril(
        #     Add), pbat.math.linalg.Cholmod.SparseStorage.SymmetricLowerTriangular)
        x = np.zeros_like(b)
        x[dofs] = Addinv.solve(bd).squeeze()
        return x


class CCD():

    def __init__(self,
                 params: Parameters,
                 broad_phase_method: ipctk.BroadPhaseMethod = ipctk.BroadPhaseMethod.HASH_GRID):
        self.params = params
        self.broad_phase_method = broad_phase_method

    def __call__(self, x: np.ndarray, dx: np.ndarray) -> float:
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        dmin = self.params.dmin
        broad_phase_method = self.broad_phase_method

        BXt0 = to_surface(x, mesh, cmesh)
        BXt1 = to_surface(x + dx, mesh, cmesh)
        max_alpha = ipctk.compute_collision_free_stepsize(
            cmesh,
            BXt0,
            BXt1,
            broad_phase_method=broad_phase_method,
            min_distance=dmin
        )
        return max_alpha


class BarrierInitializer():

    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray, gU: np.ndarray, gB: np.ndarray):
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        dhat = self.params.dhat
        dmin = self.params.dmin
        avgmass = self.params.avgmass
        bboxdiag = self.params.bboxdiag
        # Compute adaptive barrier stiffness
        BX = to_surface(x, mesh, cmesh)
        kB, maxkB = ipctk.initial_barrier_stiffness(
            bboxdiag, dhat, avgmass, gU, gB, dmin=dmin)
        dprev = cconstraints.compute_minimum_distance(cmesh, BX)
        self.params.kB = kB
        self.params.maxkB = maxkB
        self.params.dprev = dprev


class BarrierUpdater():

    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, xk: np.ndarray):
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        kB = self.params.kB
        maxkB = self.params.maxkB
        dprev = self.params.dprev
        bboxdiag = self.params.bboxdiag

        BX = to_surface(xk, mesh, cmesh)
        dcurrent = cconstraints.compute_minimum_distance(cmesh, BX)
        self.params.kB = ipctk.update_barrier_stiffness(
            dprev, dcurrent, maxkB, kB, bboxdiag, dmin=dmin)
        self.params.dprev = dcurrent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D elastic simulation of linear FEM tetrahedra using Incremental Potential Contact",
    )
    parser.add_argument(
        "-i", "--input", help="Path to input mesh", dest="input", required=True)
    parser.add_argument("-t", "--translation", help="Vertical translation", type=float,
                        dest="translation", default=0.1)
    parser.add_argument("--percent-fixed", help="Percentage of input mesh's bottom to fix", type=float,
                        dest="percent_fixed", default=0.1)
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    parser.add_argument(
        "-c", "--copy", help="Number of copies of input model", type=int, dest="ncopy", default=1)
    args = parser.parse_args()

    # Load input meshes and combine them into 1 mesh
    V, C = [], []
    imesh = meshio.read(args.input)
    V1 = imesh.points.astype(np.float64, order='C')
    C1 = imesh.cells_dict["tetra"].astype(np.int64, order='C')
    V.append(V1)
    C.append(C1)
    for c in range(args.ncopy):
        R = sp.spatial.transform.Rotation.from_quat(
            [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]).as_matrix()
        V2 = (V[-1] - V[-1].mean(axis=0)) @ R.T + V[-1].mean(axis=0)
        V2[:, 2] += (V2[:, 2].max() - V2[:, 2].min()) + args.translation
        C2 = C[-1]
        V.append(V2)
        C.append(C2)

    V, C = combine(V, C)
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    V, C = mesh.X.T, mesh.E.T

    # Construct FEM quantities for simulation
    x = mesh.X.reshape(math.prod(mesh.X.shape), order='F')
    n = x.shape[0]
    v = np.zeros(n)

    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    rho = args.rho
    M = pbat.fem.MassMatrix(mesh, detJeM, rho=rho,
                            dims=3, quadrature_order=2).to_matrix()
    # Lump mass matrix
    lumpedm = M.sum(axis=0)
    M = sp.sparse.spdiags(lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])
    Minv = sp.sparse.spdiags(
        1./lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])

    # Construct load vector from gravity field
    qgf = pbat.fem.inner_product_weights(
        mesh, quadrature_order=1).flatten(order="F")
    Qf = sp.sparse.diags_array([qgf], offsets=[0])
    Nf = pbat.fem.shape_function_matrix(mesh, quadrature_order=1)
    g = np.zeros(mesh.dims)
    g[-1] = -9.81
    fe = np.tile(rho*g[:, np.newaxis], mesh.E.shape[1])
    f = fe @ Qf @ Nf
    f = f.reshape(math.prod(f.shape), order="F")
    a = Minv @ f

    # Create hyper elastic potential
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep = pbat.fem.HyperElasticPotential(
        mesh, detJeU, GNeU, Y, nu, energy=psi, quadrature_order=1)
    hep.precompute_hessian_sparsity()

    # Setup IPC contact handling
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
    E = ipctk.edges(F)
    cmesh = ipctk.CollisionMesh.build_from_full_mesh(V, E, F)
    dhat = 1e-3
    cconstraints = ipctk.CollisionConstraints()
    fconstraints = ipctk.FrictionConstraints()
    mu = 0.3
    epsv = 1e-4
    dmin = 1e-4

    # Fix some percentage of bottom of the input models as Dirichlet boundary conditions
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    dX = Xmax - Xmin
    Xmax[-1] = Xmin[-1] + args.percent_fixed*dX[-1]
    Xmin[-1] = Xmin[-1] - 1e-4
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    dbcs = np.array(vdbc)[:, np.newaxis]
    dbcs = np.repeat(dbcs, mesh.dims, axis=1)
    for d in range(mesh.dims):
        dbcs[:, d] = mesh.dims*dbcs[:, d]+d
    dbcs = dbcs.reshape(math.prod(dbcs.shape))
    dofs = np.setdiff1d(list(range(n)), dbcs)

    # Setup GUI
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Incremental Potential Contact")
    ps.init()
    vm = ps.register_surface_mesh(
        "Visual mesh", cmesh.rest_positions, cmesh.faces)
    pc = ps.register_point_cloud("Dirichlet", mesh.X[:, vdbc].T)
    dt = 0.01
    animate = False
    newton_maxiter = 10
    newton_rtol = 1e-5

    profiler = pbat.profiling.Profiler()
    # ipctk.set_logger_level(ipctk.LoggerLevel.trace)

    def callback():
        global x, v, dt
        global dhat, dmin, mu
        global newton_maxiter, newton_rtol
        global animate, step

        changed, dt = imgui.InputFloat("dt", dt)
        changed, dhat = imgui.InputFloat(
            "IPC activation distance", dhat, format="%.6f")
        changed, dmin = imgui.InputFloat(
            "IPC minimum distance", dmin, format="%.6f")
        changed, mu = imgui.InputFloat(
            "Coulomb friction coeff", mu, format="%.2f")
        changed, newton_maxiter = imgui.InputInt(
            "Newton max iterations", newton_maxiter)
        changed, newton_rtol = imgui.InputFloat(
            "Newton convergence residual", newton_rtol, format="%.8f")
        changed, animate = imgui.Checkbox("animate", animate)
        step = imgui.Button("step")

        if animate or step:
            ps.screenshot()
            profiler.begin_frame("Physics")
            params = Parameters(mesh, x, v, a, M, hep, dt, cmesh,
                                cconstraints, fconstraints, dhat, dmin, mu, epsv)
            f = Potential(params)
            g = Gradient(params)
            H = Hessian(params)
            solver = LinearSolver(dofs)
            ccd = CCD(params)
            updater = BarrierUpdater(params)
            xtp1 = newton(x, f, g, H, solver, ccd,
                          newton_maxiter, newton_rtol, updater)
            v = (xtp1 - x) / dt
            x = xtp1
            BX = to_surface(x, mesh, cmesh)
            profiler.end_frame("Physics")

            # Update visuals
            vm.update_vertex_positions(BX)

    ps.set_user_callback(callback)
    ps.show()
