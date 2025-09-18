import pbatoolkit as pbat
import meshio
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import math
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simple 3D elastic simulation using quadratic FEM tetrahedra",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input mesh",
        type=str,
        dest="input",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--refined-input",
        help="Path to refined input mesh",
        type=str,
        dest="rinput",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mass-density",
        help="Mass density",
        type=float,
        dest="rho",
        default=1000.0,
    )
    parser.add_argument(
        "-Y",
        "--young-modulus",
        help="Young's modulus",
        type=float,
        dest="Y",
        default=1e6,
    )
    parser.add_argument(
        "-n",
        "--poisson-ratio",
        help="Poisson's ratio",
        type=float,
        dest="nu",
        default=0.45,
    )
    args = parser.parse_args()

    # Load domain and visual meshes
    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    V = V.astype(np.float64, order="c")
    C = C.astype(np.int64, order="c")
    rimesh = meshio.read(args.rinput)
    VR, FR = rimesh.points, rimesh.cells_dict["triangle"]
    VR = VR.astype(np.float64, order="c")
    FR = FR.astype(np.int64, order="c")

    # Construct FEM quantities for simulation
    element = pbat.fem.Element.Tetrahedron
    order = 2
    X, E = pbat.fem.mesh(V.T, C.T, element=element, order=order)
    dims = X.shape[0]
    x = X.reshape(math.prod(X.shape), order="f")
    v = np.zeros(x.shape[0])

    # Mass matrix
    rho = args.rho
    M = pbat.fem.mass_matrix(E, X, rho=rho, dims=dims, element=element, order=order)
    Minv = pbat.math.linalg.ldlt(M)
    Minv.compute(M)

    # Construct load vector from gravity field
    g = np.zeros(dims)
    g[-1] = -9.81
    fe = rho * g
    f = pbat.fem.load_vector(E, X, fe, element=element, order=order)
    a = Minv.solve(np.ravel(f, order="F")).squeeze()

    # Create hyper elastic potential
    Y, nu, psi = args.Y, args.nu, pbat.fem.HyperElasticEnergy.StableNeoHookean
    mu, llambda = pbat.fem.lame_coefficients(Y, nu)
    wg = pbat.fem.mesh_quadrature_weights(
        E, X, element, order=order, quadrature_order=order
    )
    eg = pbat.fem.mesh_quadrature_elements(E, wg)
    mug = np.full(math.prod(wg.shape), mu, dtype=x.dtype)
    lambdag = np.full(math.prod(wg.shape), llambda, dtype=x.dtype)
    GNegU = pbat.fem.shape_function_gradients(
        E, X, element=element, dims=dims, order=order, quadrature_order=order
    )
    ElasticityComputationFlags = pbat.fem.ElementElasticityComputationFlags
    spd_correction = pbat.fem.HyperElasticSpdCorrection.Absolute
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean

    # Set Dirichlet boundary conditions
    Xmin = X.min(axis=1)
    Xmax = X.max(axis=1)
    Xmax[0] = Xmin[0] + 1e-4
    Xmin[0] = Xmin[0] - 1e-4
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = np.array(aabb.contained(X))
    dbcs = vdbc[:, np.newaxis]
    dbcs = np.repeat(dbcs, dims, axis=1)
    for d in range(dims):
        dbcs[:, d] = dims * dbcs[:, d] + d
    dbcs = dbcs.reshape(math.prod(dbcs.shape))
    n = x.shape[0]
    dofs = np.setdiff1d(list(range(n)), dbcs)

    # Setup linear solver
    U, _, H = pbat.fem.hyper_elastic_potential(
        E,
        X.shape[1],
        eg=np.ravel(eg, order="F"),
        wg=np.ravel(wg, order="F"),
        GNeg=GNegU,
        mug=mug,
        lambdag=lambdag,
        x=x,
        energy=energy,
        flags=ElasticityComputationFlags.Potential | ElasticityComputationFlags.Hessian,
        spd_correction=spd_correction,
        element=element,
        order=order,
        dims=dims,
    )
    print("Rest energy U={}".format(U))
    Hdd = H.tocsc()[:, dofs].tocsr()[dofs, :]
    Hddinv = pbat.math.linalg.ldlt(Hdd)
    Hddinv.analyze(Hdd)

    # Setup visual mesh
    bvh = pbat.geometry.bvh(V.T, C.T, cell=pbat.geometry.Cell.Tetrahedron)
    e, d = bvh.nearest_primitives_to_points(VR.T)
    Xi = pbat.fem.reference_positions(E, X, e, VR.T, element=element, order=order)
    phi = pbat.fem.shape_functions_at(Xi, element=element, order=order)

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Higher Order Elasticity")
    ps.init()
    vm = ps.register_volume_mesh("Domain", V, C)
    sm = ps.register_surface_mesh("Visual", VR, FR)
    pc = ps.register_point_cloud("Dirichlet", X[:, vdbc].T)
    dt = 0.01
    animate = False
    dx = np.zeros(n)

    profiler = pbat.profiling.Profiler()

    def callback():
        global x, v, dx, hep, dt, M, Minv, f, animate, step, profiler
        changed, dt = imgui.InputFloat("dt", dt)
        changed, animate = imgui.Checkbox("animate", animate)
        step = imgui.Button("step")

        if animate or step:
            profiler.begin_frame("Physics")
            # 1 Newton step
            _, gradU, HU = pbat.fem.hyper_elastic_potential(
                E,
                X.shape[1],
                eg=np.ravel(eg, order="F"),
                wg=np.ravel(wg, order="F"),
                GNeg=GNegU,
                mug=np.ravel(mug, order="F"),
                lambdag=np.ravel(lambdag, order="F"),
                x=np.ravel(x, order="F"),
                energy=energy,
                flags=ElasticityComputationFlags.Gradient
                | ElasticityComputationFlags.Hessian,
                spd_correction=spd_correction,
                element=element,
                order=order,
                dims=dims,
            )
            dt2 = dt**2
            xtilde = x + dt * v + dt2 * a
            A = M + dt2 * HU
            b = -(M @ (x - xtilde) + dt2 * gradU)
            Add = A.tocsc()[:, dofs].tocsr()[dofs, :]
            bd = b[dofs]
            Hddinv.factorize(Add)
            dx[dofs] = Hddinv.solve(bd).squeeze()
            v = dx / dt
            x = x + dx
            profiler.end_frame("Physics")

            # Update visuals
            V = x.reshape(X.shape[0], X.shape[1], order="f")
            xv = (V[:, E[:, e]] * phi).sum(axis=1)
            sm.update_vertex_positions(xv.T)

    ps.set_user_callback(callback)
    ps.show()
