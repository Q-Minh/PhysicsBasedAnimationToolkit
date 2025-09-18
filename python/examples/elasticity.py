import pbatoolkit as pbat
import ilupp
import meshio
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import math
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simple 3D elastic simulation using linear FEM tetrahedra",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output", type=str,
                        dest="output", default=".")
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    args = parser.parse_args()

    # Construct FEM quantities for simulation
    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    element = pbat.fem.Element.Tetrahedron
    X, E = pbat.fem.mesh(
        V.T, C.T, element=element, order=1)
    x = X.reshape(math.prod(X.shape), order='f')
    v = np.zeros(x.shape[0])
    dims = X.shape[0]
    order=1

    # Mass matrix
    rho = args.rho
    M = pbat.fem.mass_matrix(E, X, rho=rho, dims=dims, element=element, order=order)
    Minv = pbat.math.linalg.ldlt(M)
    Minv.compute(M)

    # Construct load vector from gravity field
    g = np.zeros(dims)
    g[-1] = -9.81
    fe = rho*g
    f = pbat.fem.load_vector(E, X, fe, element=element, order=order)
    a = Minv.solve(f.flatten(order="F")).squeeze()

    # Create hyper elastic potential
    Y, nu, energy = args.Y, args.nu, pbat.fem.HyperElasticEnergy.StableNeoHookean
    mu, llambda = pbat.fem.lame_coefficients(Y, nu)
    mug = np.full(E.shape[1], mu, dtype=x.dtype)
    lambdag = np.full(E.shape[1], llambda, dtype=x.dtype)
    eg = np.arange(E.shape[1], dtype=np.int32)
    GNegU = pbat.fem.shape_function_gradients(
        E, X, element=element, dims=dims, order=order
    )
    wg = pbat.fem.mesh_quadrature_weights(E, X, element, order=order, quadrature_order=1).flatten()
    ElasticityComputationFlags = pbat.fem.ElementElasticityComputationFlags
    spd_correction = pbat.fem.HyperElasticSpdCorrection.Absolute

    # Set Dirichlet boundary conditions
    Xmin = X.min(axis=1)
    Xmax = X.max(axis=1)
    Xmax[0] = Xmin[0]+1e-4
    Xmin[0] = Xmin[0]-1e-4
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = np.array(aabb.contained(X))
    dbcs = vdbc[:, np.newaxis]
    dbcs = np.repeat(dbcs, dims, axis=1)
    for d in range(dims):
        dbcs[:, d] = dims*dbcs[:, d]+d
    dbcs = dbcs.reshape(math.prod(dbcs.shape))
    n = x.shape[0]
    dofs = np.setdiff1d(list(range(n)), dbcs)

    # Setup linear solver
    _, _, H = pbat.fem.hyper_elastic_potential(
        E,
        X.shape[1],
        eg=eg,
        wg=wg,
        GNeg=GNegU,
        mug=mug,
        lambdag=lambdag,
        x=x,
        energy=energy,
        flags=ElasticityComputationFlags.Hessian,
        spd_correction=spd_correction,
        element=element,
        order=order,
        dims=dims
    )
    Hdd = H.tocsc()[:, dofs].tocsr()[dofs, :]
    Mdd = M.tocsc()[:, dofs].tocsr()[dofs, :]
    Addinv = pbat.math.linalg.ldlt(
        Hdd, solver=pbat.math.linalg.SolverBackend.Eigen)
    Addinv.analyze(Hdd)

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Elasticity")
    ps.init()
    vm = ps.register_volume_mesh("world model", X.T, E.T)
    pc = ps.register_point_cloud("Dirichlet", X[:, vdbc].T)
    dt = 0.01
    animate = False
    use_direct_solver = False
    export = False
    t = 0
    newton_maxiter = 1
    cg_fill_in = 0.01
    cg_drop_tolerance = 1e-4
    cg_residual = 1e-5
    cg_maxiter = 100
    dx = np.zeros(n)

    profiler = pbat.profiling.Profiler()

    def callback():
        global x, v, dx, hep, dt, M, f
        global X, E, eg, mug, lambdag, wg, GNegU, energy, element, order, dims
        global cg_fill_in, cg_drop_tolerance, cg_residual, cg_maxiter
        global animate, step, use_direct_solver, export, t
        global newton_maxiter
        global profiler

        changed, dt = imgui.InputFloat("dt", dt)
        changed, newton_maxiter = imgui.InputInt(
            "Newton max iterations", newton_maxiter)
        changed, cg_fill_in = imgui.InputFloat(
            "IC column fill in", cg_fill_in, format="%.4f")
        changed, cg_drop_tolerance = imgui.InputFloat(
            "IC drop tolerance", cg_drop_tolerance, format="%.8f")
        changed, cg_residual = imgui.InputFloat(
            "PCG residual", cg_residual, format="%.8f")
        changed, cg_maxiter = imgui.InputInt(
            "PCG max iterations", cg_maxiter)
        changed, animate = imgui.Checkbox("animate", animate)
        changed, use_direct_solver = imgui.Checkbox(
            "Use direct solver", use_direct_solver)
        changed, export = imgui.Checkbox("Export", export)
        step = imgui.Button("step")

        if animate or step:
            profiler.begin_frame("Physics")
            # Newton solve
            dt2 = dt**2
            xtilde = x + dt*v + dt2*a
            xk = x
            for k in range(newton_maxiter):
                _, gradU, HU = pbat.fem.hyper_elastic_potential(
                    E,
                    X.shape[1],
                    eg=eg,
                    wg=wg,
                    GNeg=GNegU,
                    mug=mug,
                    lambdag=lambdag,
                    x=xk,
                    energy=energy,
                    flags=ElasticityComputationFlags.Gradient | ElasticityComputationFlags.Hessian,
                    spd_correction=spd_correction,
                    element=element,
                    order=order,
                    dims=dims
                )

                global bd, Add

                def setup():
                    global bd, Add
                    A = M + dt2 * HU
                    b = -(M @ (xk - xtilde) + dt2*gradU)
                    Add = A.tocsc()[:, dofs].tocsr()[dofs, :]
                    bd = b[dofs]

                profiler.profile("Setup Linear System", setup)

                if k > 0:
                    gradnorm = np.linalg.norm(bd, 1)
                    if gradnorm < 1e-3:
                        break

                def solve():
                    global dx, Add, bd
                    global cg_fill_in, cg_drop_tolerance, cg_maxiter, cg_residual
                    global use_direct_solver
                    if use_direct_solver:
                        Addinv.factorize(Add)
                        dx[dofs] = Addinv.solve(bd).squeeze()
                    else:
                        P = ilupp.ICholTPreconditioner(
                            Add, add_fill_in=int(Add.shape[0]*cg_fill_in), threshold=cg_drop_tolerance)
                        dx[dofs], cginfo = sp.sparse.linalg.cg(
                            Add, bd, rtol=cg_residual, maxiter=cg_maxiter, M=P)

                profiler.profile("Solve Linear System", solve)
                xk = xk + dx

            v = (xk - x) / dt
            x = xk
            profiler.end_frame("Physics")

            if export:
                ps.screenshot(f"{args.output}/{t}.png")

            # Update visuals
            V = x.reshape(X.shape[0], X.shape[1], order='f')
            vm.update_vertex_positions(V.T)

            t = t + 1

        imgui.Text(f"Frame={t}")

    ps.set_user_callback(callback)
    ps.show()
