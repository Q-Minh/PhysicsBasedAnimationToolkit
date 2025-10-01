import pbatoolkit as pbat
import polyscope as ps
import polyscope.imgui as imgui
import polyscope.implot as implot
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections.abc import Callable
from typing import Tuple


def minimize_quadratic(a, b, c):
    """
    Return the minimizer x in [0,1] of q(x) = c x^2 + b x + a.
    """
    if c > 0:
        tstar = -b / (2 * c)
        tstar = min(max(tstar, 0.0), 1.0)
    else:
        if b > 0:
            tstar = 0.0
        else:
            tstar = 1.0
    return tstar


def solve_quadratic_in_reference_triangle_2d(xk, gk, Bk) -> np.ndarray:
    xstar = xk - np.linalg.solve(Bk, gk)
    feasible = (xstar >= 0).all() and (xstar <= 1).all() and (xstar.sum() <= 1)
    if not feasible:
        # NOTE: Uncomment to validate against symbolic derivation
        # if it ever happens that pred < 0 (should never happen).
        # alpha0 = np.zeros(2)
        # dalpha = np.array([0, 1])
        # a1 = gk.T @ (alpha0 - xk) + 0.5 * (alpha0 - xk).T @ Bk @ (alpha0 - xk)
        # b1 = gk.T @ dalpha + dalpha.T @ Bk @ (alpha0 - xk)
        # c1 = 0.5 * dalpha.T @ Bk @ dalpha
        # tmin1 = minimize_quadratic(a1, b1, c1)
        # alpha0 = np.zeros(2)
        # dalpha = np.array([1, 0])
        # a2 = gk.T @ (alpha0 - xk) + 0.5 * (alpha0 - xk).T @ Bk @ (alpha0 - xk)
        # b2 = gk.T @ dalpha + dalpha.T @ Bk @ (alpha0 - xk)
        # c2 = 0.5 * dalpha.T @ Bk @ dalpha
        # tmin2 = minimize_quadratic(a2, b2, c2)
        # alpha0 = np.array([0, 1])
        # dalpha = np.array([1, -1])
        # a3 = gk.T @ (alpha0 - xk) + 0.5 * (alpha0 - xk).T @ Bk @ (alpha0 - xk)
        # b3 = gk.T @ dalpha + dalpha.T @ Bk @ (alpha0 - xk)
        # c3 = 0.5 * dalpha.T @ Bk @ dalpha
        # tmin3 = minimize_quadratic(a3, b3, c3)

        # Derivation and CSE by-hand for the quadratic
        # f(t) = 0.5 (x0 + t dx - xk)^T Bk (x0 + t dx - xk) + gk^T (x0 + t dx - xk)
        # where x = x0 + t dx is constrained to one of the triangle edges.
        # Edge 1: x0 = [0,0], dx = [0,1]
        # Edge 2: x0 = [0,0], dx = [1,0]
        # Edge 3: x0 = [0,1], dx = [1,-1]
        gkTxk = gk.T @ xk
        xkTBkxk = xk.T @ Bk @ xk
        Bkxk = Bk @ xk
        a1 = -gkTxk + 0.5 * xkTBkxk
        b1 = gk[1] - Bkxk[1]
        c1 = Bk[1, 1]
        a2 = a1
        b2 = gk[0] - Bkxk[0]
        c2 = Bk[0, 0]
        xk0 = np.array([-xk[0], 1 - xk[1]])
        a3 = gk.T @ xk0 + 0.5 * xk0.T @ Bk @ xk0
        b3 = (gk[0] - gk[1]) + (Bk[0, 1] - Bk[1, 1]) - (Bkxk[0] - Bkxk[1])
        c3 = Bk[0, 0] - 2 * Bk[0, 1] + Bk[1, 1]
        # Minimize quadratic assuming c_i > 0
        tmin1 = min(max(-b1 / c1, 0.0), 1.0)
        tmin2 = min(max(-b2 / c2, 0.0), 1.0)
        tmin3 = min(max(-b3 / c3, 0.0), 1.0)
        fmins = [
            a1 + b1 * tmin1 + 0.5 * c1 * tmin1**2,
            a2 + b2 * tmin2 + 0.5 * c2 * tmin2**2,
            a3 + b3 * tmin3 + 0.5 * c3 * tmin3**2,
        ]
        imin = np.argmin(fmins)
        if imin == 0:
            xstar = np.array([0.0, tmin1])
        elif imin == 1:
            xstar = np.array([tmin2, 0.0])
        else:
            xstar = np.array([tmin3, 1.0 - tmin3])
    return xstar


def step_minimize_triangle(
    f: Callable[[np.ndarray], float],
    g: Callable[[np.ndarray], np.ndarray],
    xk: np.ndarray,
    fk: float,
    gk: np.ndarray,
    Bk: np.ndarray,
    Rk: float,
    eta: float = 1e-3,
    r: float = 1e-8,
    trlo: float = 0.1,
    trhi: float = 0.75,
    trbound: float = 0.8,
    trgrow: float = 2.0,
    trshrink: float = 0.5,
    eps: float = 1e-4,
) -> Tuple[
    np.ndarray, float, np.ndarray, np.ndarray, float
]:  # xk+1, fk+1, gk+1, Bk+1, Rk+1
    xkp1 = solve_quadratic_in_reference_triangle_2d(xk, gk, Bk)
    sk = xkp1 - xk
    # Use max-norm, i.e. box trust region
    # if np.dot(sk, sk) > Rk * Rk:
    #     sk = sk * Rk / np.linalg.norm(sk)
    lensk = np.linalg.norm(sk, np.inf)
    if lensk > Rk:
        sk = sk * Rk / lensk
        xkp1 = xk + sk
        lensk = np.linalg.norm(sk, np.inf)
    gkp1 = g(xkp1)
    yk = gkp1 - gk
    fkp1 = f(xkp1)
    ared = fk - fkp1
    mkp1 = gk.T @ sk + 0.5 * sk.T @ Bk @ sk
    pred = -mkp1
    rho = ared / (pred + eps)
    Rkp1 = Rk
    if rho > trhi and lensk <= trbound * Rk:
        Rkp1 = trgrow * Rk
    elif rho < trlo:
        Rkp1 = trshrink * Rk
    vk = yk - Bk @ sk
    den = np.dot(vk, sk)
    # stable = den**2 >= r * np.dot(sk, sk) * np.dot(vk, vk)
    # Bkp1 = Bk + np.outer(vk, vk) / den if stable else Bk
    # Update inverse hessian estimate and keep positive definite
    Bkp1 = Bk
    skTyk = np.dot(sk, yk)
    skTBksk = np.dot(sk, Bk @ sk)
    if skTyk > skTBksk:
        Bkp1 = Bk + np.outer(vk, vk) / den
    if rho <= eta:
        xkp1 = xk
        fkp1 = fk
        gkp1 = gk
    return xkp1, fkp1, gkp1, Bkp1, Rkp1


if __name__ == "__main__":
    # Domain
    extent = 1
    bmin = -extent * np.ones(3)
    bmax = extent * np.ones(3)
    dims = (100, 100, 100)
    # polyscope's volume grid expects x to vary fastest, then y, then z
    x, y, z = np.meshgrid(
        np.linspace(bmin[0], bmax[0], dims[0]),
        np.linspace(bmin[1], bmax[1], dims[1]),
        np.linspace(bmin[2], bmax[2], dims[2]),
        indexing="ij",
    )
    X = np.vstack([np.ravel(z), np.ravel(y), np.ravel(x)]).astype(np.float64)

    # Triangle
    V = np.array(
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    F = np.array([[0, 1, 2]])

    # SDF
    forest = pbat.geometry.sdf.Forest()

    # Optimization
    xk = np.zeros(2)
    fk = 0.0
    gk = np.zeros(2)
    Bk = np.eye(2)
    Rk = 1.0
    sigma = 1.0
    eta = 1e-3
    r = 1e-8
    trlo = 0.1
    trhi = 0.75
    trbound = 0.8
    trgrow = 2.0
    trshrink = 0.5

    xpath = []
    fpath = []

    # Polyscope visualization
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("SDF editor")
    ps.init()

    slice_plane = ps.add_scene_slice_plane()
    slice_plane.set_draw_plane(False)
    slice_plane.set_draw_widget(True)
    isolines = True
    enable_isosurface_viz = True
    isoline_contour_thickness = 0.3
    vminmax = (-extent, extent)
    cmap = "coolwarm"
    grid = ps.register_volume_grid("Domain", dims, bmin, bmax)
    grid.set_transparency(0.75)
    sm = ps.register_surface_mesh("Triangle", V, F)
    sm.set_ignore_slice_plane(slice_plane, True)

    def callback():
        global forest
        global xk, fk, gk, Bk, Rk, sigma
        global eta, r, trlo, trhi, trbound, trgrow, trshrink
        global xpath, fpath

        # Load
        if imgui.TreeNode("I/O"):
            if imgui.Button("Load", [imgui.GetWindowWidth() / 2.1, 0]):
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select SDF forest file",
                    defaultextension=".h5",
                    filetypes=[("SDF forest files", "*.h5"), ("All files", "*.*")],
                )
                if file_path:
                    archive = pbat.io.Archive(file_path, pbat.io.AccessMode.ReadOnly)
                    forest.deserialize(archive)
                    composite = pbat.geometry.sdf.Composite(forest)
                    sd_composite = composite.eval(X).reshape(dims, order="F")
                    grid.add_scalar_quantity(
                        "SDF",
                        sd_composite,
                        defined_on="nodes",
                        cmap=cmap,
                        vminmax=vminmax,
                        isolines_enabled=isolines,
                        # isoline_contour_thickness=isoline_contour_thickness,
                        enable_isosurface_viz=enable_isosurface_viz,
                        enabled=True,
                    )
                root.destroy()
            imgui.TreePop()

        # Optimize
        if imgui.TreeNode("Optimize"):
            # Parameters
            changed, sigma = imgui.SliderFloat("sigma", sigma, 1e-6, 1e2)
            changed, eta = imgui.SliderFloat("eta", eta, 1e-6, 0.5)
            changed, r = imgui.SliderFloat("r", r, 1e-10, 1e-1)
            changed, trlo = imgui.SliderFloat("trlo", trlo, 1e-2, 0.9)
            changed, trhi = imgui.SliderFloat("trhi", trhi, 1e-2, 0.9)
            changed, trbound = imgui.SliderFloat("trbound", trbound, 0.5, 1.0)
            changed, trgrow = imgui.SliderFloat("trgrow", trgrow, 1.1, 10.0)
            changed, trshrink = imgui.SliderFloat("trshrink", trshrink, 1e-2, 0.99)

            # Triangle
            VH = np.vstack([V.T, np.ones((1, V.shape[0]))])
            T = sm.get_transform()
            ABC = (T @ VH)[:3, :]
            A, B, C = ABC[:, 0], ABC[:, 1], ABC[:, 2]
            DX = np.vstack([B - A, C - A]).T

            # Objective
            sdf = pbat.geometry.sdf.Composite(forest)

            def f(x: np.ndarray) -> float:
                return sdf.eval(DX @ x + A)

            def g(x: np.ndarray) -> np.ndarray:
                h = 1e-4
                gx = sdf.grad(DX @ x + A, h)
                return DX.T @ gx

            # Controls
            if imgui.Button("Step"):
                xkp1, fkp1, gkp1, Bkp1, Rkp1 = step_minimize_triangle(
                    f,
                    g,
                    xk,
                    fk,
                    gk,
                    Bk,
                    Rk,
                    eta,
                    r,
                    trlo,
                    trhi,
                    trbound,
                    trgrow,
                    trshrink,
                )
                if np.linalg.norm(xk - xkp1) > 0.0:
                    xpath = xpath + [DX @ xkp1 + A]
                    fpath = fpath + [fkp1]
                xk, fk, gk, Bk, Rk = xkp1, fkp1, gkp1, Bkp1, Rkp1
            if imgui.Button("Reset"):
                xk = np.array([0.25, 0.25])
                fk = f(xk)
                gk = g(xk)
                elen = [
                    np.linalg.norm(A - B),
                    np.linalg.norm(A - C),
                    np.linalg.norm(B - C),
                ]
                Bk = np.eye(2) * sigma * max(elen)
                Rk = max(elen)
                xpath = [DX @ xk + A]
                fpath = [fk]

            if len(xpath) > 0:
                VE = np.array(xpath)
                EE = np.vstack([np.arange(len(xpath) - 1), np.arange(1, len(xpath))]).T
                cn = ps.register_curve_network(
                    "Optimization Path",
                    VE,
                    EE,
                )
                pc = ps.register_point_cloud("Current Point", VE[-1:, :])
                cn.set_ignore_slice_plane(slice_plane, True)
                pc.set_ignore_slice_plane(slice_plane, True)
                pc.set_radius(1.1 * cn.get_radius(), relative=False)

            if len(fpath) > 0:
                if implot.BeginPlot("Objective Value"):
                    implot.PlotLine(
                        "sdf", np.array(fpath), 1 / len(fpath), 0.0  # xscale  # xstart
                    )
                    implot.EndPlot()

            imgui.TreePop()

    ps.set_user_callback(callback)
    ps.show()
