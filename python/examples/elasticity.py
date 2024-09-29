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
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    x = mesh.X.reshape(math.prod(mesh.X.shape), order='f')
    v = np.zeros(x.shape[0])
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    rho = args.rho
    M = pbat.fem.MassMatrix(mesh, detJeM, rho=rho,
                            dims=3, quadrature_order=2).to_matrix()
    Minv = pbat.math.linalg.ldlt(M)
    Minv.compute(M)
    # Could also lump the mass matrix like this
    # lumpedm = M.sum(axis=0)
    # M = sp.sparse.spdiags(lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])
    # Minv = sp.sparse.spdiags(
    #     1./lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])

    # Construct load vector from gravity field
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    qgf = pbat.fem.inner_product_weights(
        mesh, quadrature_order=1).flatten(order="F")
    Qf = sp.sparse.diags_array([qgf], offsets=[0])
    Nf = pbat.fem.shape_function_matrix(mesh, quadrature_order=1)
    g = np.zeros(mesh.dims)
    g[-1] = -9.81
    fe = np.tile(rho*g[:, np.newaxis], mesh.E.shape[1])
    f = fe @ Qf @ Nf
    f = f.reshape(math.prod(f.shape), order="F")
    a = Minv.solve(f).squeeze()

    # Create hyper elastic potential
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep = pbat.fem.HyperElasticPotential(
        mesh, detJeU, GNeU, Y, nu, energy=psi, quadrature_order=1)
    hep.precompute_hessian_sparsity()

    # Set Dirichlet boundary conditions
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    Xmax[0] = Xmin[0]+1e-4
    Xmin[0] = Xmin[0]-1e-4
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    dbcs = np.array(vdbc)[:, np.newaxis]
    dbcs = np.repeat(dbcs, mesh.dims, axis=1)
    for d in range(mesh.dims):
        dbcs[:, d] = mesh.dims*dbcs[:, d]+d
    dbcs = dbcs.reshape(math.prod(dbcs.shape))
    n = x.shape[0]
    dofs = np.setdiff1d(list(range(n)), dbcs)

    # Setup linear solver
    Hdd = hep.to_matrix()[:, dofs].tocsr()[dofs, :]
    Mdd = M[:, dofs].tocsr()[dofs, :]
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
    vm = ps.register_volume_mesh("world model", mesh.X.T, mesh.E.T)
    pc = ps.register_point_cloud("Dirichlet", mesh.X[:, vdbc].T)
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
        global x, v, dx, hep, dt, M, Minv, f
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
                hep.compute_element_elasticity(xk, grad=True, hessian=True)
                gradU, HU = hep.gradient(), hep.hessian()

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
            X = x.reshape(mesh.X.shape[0], mesh.X.shape[1], order='f')
            vm.update_vertex_positions(X.T)

            t = t + 1

        imgui.Text(f"Frame={t}")

    ps.set_user_callback(callback)
    ps.show()
