from pbatoolkit import pbat
import polyscope as ps
import polyscope.imgui as imgui
import polyscope.implot as implot
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections.abc import Callable
from typing import Tuple
import json
import argparse

def step_adam(g: Callable[[np.ndarray], np.ndarray],xk: np.ndarray, mt: np.ndarray, vt: float, beta1: float, beta2: float, eta: float, t: int, eps: float = 1e-4,) -> Tuple[np.ndarray, np.ndarray, float]:
    grad = g(xk)

    # projected gradient workaround
    xk_tmp = step_proj_gradient_decent(g,xk,eta)
    proj_grad = (xk - xk_tmp) / eta

    mt_next = mt * beta1 + (1- beta1) * proj_grad
    vt_next = beta2 * vt + (1-beta2) * np.dot(proj_grad,proj_grad)

    mt_hat = mt_next / (1-beta1**t)
    vt_hat = vt_next / (1-beta2**t)

    xkp1 = xk - eta * mt_hat / (np.sqrt(vt_hat) + eps)

    return (xkp1, mt_next, vt_next)



def step_frank_wolfe(g: Callable[[np.ndarray], np.ndarray],xk: np.ndarray, vertices: np.ndarray, t: int) -> np.ndarray:
    ''' use space coordinates'''

    grad = g(xk)

    d = vertices.T @ grad
    # min_v = sorted(zip(vertices,d),key=lambda el: el[1])[0][0]
    min_v = vertices[:,np.argmin(d)]
    alpha = 2./(t + 2)
    xkp1 = xk + alpha * (min_v - xk)

    return xkp1
    

def step_proj_gradient_decent(g: Callable[[np.ndarray], np.ndarray],xk: np.ndarray,eta: float) -> np.ndarray:
    gk = g(xk)
    xkp1 = xk - eta * gk
    xkp1[0] = 0. if xkp1[0] < 0. else xkp1[0]
    xkp1[1] = 0. if xkp1[1] < 0. else xkp1[1]

    # projection might be wrong
    if xkp1[0] + xkp1[1] > 1:
        # xkp1 /= xkp1[0] + xkp1[1]
        alpha = np.array([1.,0.])
        v = np.array([-1.,1.])
        x = xkp1 - alpha
        t  = np.clip(np.dot(v,x)/np.dot(v,v),0.,1.)
        xkp1 = np.array(alpha) + t * v

    return xkp1


def sample_in_reference_triangle_2d() -> np.ndarray:
    u = np.random.rand()
    v = np.random.rand()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    return np.array([u, v])


def solve_quadratic_in_reference_triangle_2d(xk, gk, Bk) -> np.ndarray:
    xstar = xk - np.linalg.solve(Bk, gk)
    feasible = (xstar >= 0).all() and (xstar <= 1).all() and (xstar.sum() <= 1)
    if not feasible:
        # Derivation and CSE by-hand for the quadratic
        # f(t) = 0.5 (x0 + t dx - xk)^T Bk (x0 + t dx - xk) + gk^T (x0 + t dx - xk)
        # where x = x0 + t dx is constrained to one of the triangle edges.
        # Edge 1: x0 = [0,0], dx = [0,1]
        # Edge 2: x0 = [0,0], dx = [1,0]
        # Edge 3: x0 = [0,1], dx = [1,-1]
        gkTxk = gk.T @ xk
        Bkxk = Bk @ xk
        xkTBkxk = xk.T @ Bkxk
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
        # Minimize quadratic a_i + b_i t + 1/2 c_i t^2 assuming c_i > 0
        tmin1 = min(max(-b1 / c1, 0.0), 1.0)
        tmin2 = min(max(-b2 / c2, 0.0), 1.0)
        tmin3 = min(max(-b3 / c3, 0.0), 1.0)
        fmins = [
            a1 + b1 * tmin1 + 0.5 * c1 * tmin1**2,
            a2 + b2 * tmin2 + 0.5 * c2 * tmin2**2,
            a3 + b3 * tmin3 + 0.5 * c3 * tmin3**2,
        ]
        # Vectorized argmin
        argmin = np.array(
            [
                fmins[0] <= fmins[1] and fmins[0] <= fmins[2],
                fmins[1] <= fmins[0] and fmins[1] <= fmins[2],
                fmins[2] <= fmins[0] and fmins[2] <= fmins[1],
            ]
        )
        xstars = np.array(
            [
                [0.0, tmin1],
                [tmin2, 0.0],
                [tmin3, 1.0 - tmin3],
            ]
        )
        xstar = xstars.T @ argmin / argmin.sum()
        # Non-vectorized argmin
        # imin = np.argmin(fmins)
        # if imin == 0:
        #     xstar = np.array([0.0, tmin1])
        # elif imin == 1:
        #     xstar = np.array([tmin2, 0.0])
        # else:
        #     xstar = np.array([tmin3, 1.0 - tmin3])
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
    # NOTE: Uncomment to use 2-norm, i.e. ball trust region
    # if np.dot(sk, sk) > Rk * Rk:
    #     sk = sk * Rk / np.linalg.norm(sk)
    # Use max-norm, i.e. box trust region
    lensk = np.linalg.norm(sk, np.inf)
    # TODO: Vectorize this branch
    if lensk > Rk:
        sk = sk * Rk / lensk
        xkp1 = xk + sk
        lensk = np.linalg.norm(sk, np.inf)
    gkp1 = g(xkp1)
    yk = gkp1 - gk
    fkp1 = f(xkp1)
    ared = fk - fkp1
    Bksk = Bk @ sk
    skTBksk = sk.T @ Bksk
    mkp1 = gk.T @ sk + 0.5 * skTBksk
    pred = -mkp1
    rho = ared / (pred + eps)
    Rkp1 = Rk
    # TODO: Vectorize these branches
    if rho > trhi and lensk >= trbound * Rk:
        Rkp1 = trgrow * Rk
    elif rho < trlo:
        Rkp1 = trshrink * Rk
    vk = yk - Bksk
    den = np.dot(vk, sk)
    # stable = den**2 >= r * np.dot(sk, sk) * np.dot(vk, vk)
    # Bkp1 = Bk + np.outer(vk, vk) / den if stable else Bk
    # Update inverse hessian estimate and keep positive definite
    Bkp1 = Bk
    skTyk = np.dot(sk, yk)
    # NOTE: For a GPU implementation, we probably also want to
    # vectorize these branches
    if skTyk > skTBksk:
        Bkp1 = Bk + np.outer(vk, vk) / den
    if rho <= eta:
        xkp1 = xk
        fkp1 = fk
        gkp1 = gk
    return xkp1, fkp1, gkp1, Bkp1, Rkp1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SDF editor",
    )
    parser.add_argument(
        "--grid-dims",
        type=int,
        nargs=3,
        metavar=("nx", "ny", "nz"),
        help="Grid dimensions as three integers",
        dest="dims",
        default=(100, 100, 100),
    )
    parser.add_argument(
        "--b",
        type=float,
        nargs=3,
        metavar=("bx", "by", "bz"),
        help="Axis-aligned grid's lower bound",
        dest="b",
        default=(-1.0, -1.0, -1.0)
    )
    parser.add_argument(
        "--e",
        type=float,
        nargs=3,
        metavar=("ex", "ey", "ez"),
        help="Axis-aligned grid's upper bound",
        dest="e",
        default=(1.0, 1.0, 1.0)
    )
    args = parser.parse_args()

    # Domain
    bmin = np.array(args.b)
    bmax = np.array(args.e)
    dims = args.dims
    extent = np.max(bmax - bmin)
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
    sigmaR = 1e-1
    sigmaB = 1e-1
    eta = 1e-3
    r = 1e-8
    trlo = 0.1
    trhi = 0.75
    trbound = 0.8
    trgrow = 2.0
    trshrink = 0.5

    xpath = []
    fpath = []
    Rpath = []

    gd_xk = np.zeros(2)
    gd_eta = 1e-3
    gd_xpath = []
    gd_fpath = []

    fw_xk = np.zeros(3)
    fw_xpath = []
    fw_fpath = []

    ad_xk = np.zeros(2)
    ad_mt = np.zeros(2)
    ad_vt = 0.
    ad_beta1 = 0.9
    ad_beta2 = 0.999
    ad_eta = 1e-3
    ad_xpath = []
    ad_fpath = []

    # For reproducibility
    np.random.seed(0)
    randomize_sample = False

    # Polyscope visualization
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("SDF editor")
    ps.init()

    slice_plane = ps.add_scene_slice_plane()
    slice_plane.set_pose([0.0, 0.0, 0.5], [0.0, 0.0, -1.0])
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
        global xk, fk, gk, Bk, Rk, sigmaB, sigmaR
        global eta, r, trlo, trhi, trbound, trgrow, trshrink
        global xpath, fpath, Rpath
        global randomize_sample
        global gd_xk
        global gd_eta
        global gd_xpath, gd_fpath
        global fw_xk
        global fw_xpath, fw_fpath
        global ad_xk, ad_mt, ad_vt
        global ad_beta1, ad_beta2, ad_eta
        global ad_xpath, ad_fpath

        # Load
        if imgui.TreeNode("I/O"):
            if imgui.Button("Load SDF", [imgui.GetWindowWidth() / 2.1, 0]):
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
            if imgui.Button("Load Parameters", [imgui.GetWindowWidth() / 2.1, 0]):
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select parameter",
                    defaultextension=".json",
                    filetypes=[("JSON", "*.json"), ("All files", "*.*")],
                )
                if file_path:
                    with open(file_path) as f_bar:
                        params = json.load(f_bar)
                        sigmaR = params.get("sigmaR",sigmaR)
                        sigmaB = params.get("sigmaB",sigmaB)
                        eta = params.get("eta",eta)
                        r = params.get("r",r)
                        trlo = params.get("trlo",trlo)
                        trhi = params.get("trhi",trhi)
                        trbound = params.get("trbound",trbound)
                        trgrow = params.get("trgrow",trgrow)
                        trshrink = params.get("trshrink",trshrink)
                        gd_eta = params.get("gd_eta",gd_eta)
                        ad_beta1 = params.get("ad_beta1",ad_beta1)
                        ad_beta2 = params.get("ad_beta2",ad_beta2)
                        ad_eta = params.get("ad_eta",ad_eta)
                    
                root.destroy()
            if imgui.Button("Save Parameters", [imgui.GetWindowWidth() / 2.1, 0]):
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.asksaveasfilename(
                    title="parameters",
                    defaultextension=".json",
                    filetypes=[("JSON", "*.json"), ("All files", "*.*")],
                )
                if file_path:
                    with open(file_path,'w') as f_bar:
                        params = {
                            "sigmaR": sigmaR,
                            "sigmaB": sigmaB,
                            "eta": eta,
                            "r": r,
                            "trlo": trlo,
                            "trhi": trhi,
                            "trbound": trbound,
                            "trgrow": trgrow,
                            "trshrink": trshrink,
                            "gd_eta": gd_eta,
                            "ad_beta1": ad_beta1,
                            "ad_beta2": ad_beta2,
                            "ad_eta": ad_eta,
                        }
                        json.dump(params,f_bar)
                    
                root.destroy()
            imgui.TreePop()

        # Optimize
        if imgui.TreeNode("Optimize"):
            # Parameters
            imgui.TextUnformatted("New method")
            changed, sigmaR = imgui.SliderFloat("sigmaR", sigmaR, 1e-2, 1e2)
            changed, sigmaB = imgui.SliderFloat("sigmaB", sigmaB, 1e-6, 1e2)
            changed, eta = imgui.SliderFloat("eta", eta, 1e-6, 0.5)
            changed, r = imgui.SliderFloat("r", r, 1e-10, 1e-1)
            changed, trlo = imgui.SliderFloat("trlo", trlo, 1e-2, 0.99)
            changed, trhi = imgui.SliderFloat("trhi", trhi, 1e-2, 1.0)
            changed, trbound = imgui.SliderFloat("trbound", trbound, 0.1, 1.0)
            changed, trgrow = imgui.SliderFloat("trgrow", trgrow, 1.1, 10.0)
            changed, trshrink = imgui.SliderFloat("trshrink", trshrink, 1e-2, 0.99)
            imgui.TextUnformatted("Gradient Descent")
            changed, gd_eta = imgui.SliderFloat("gd_eta", gd_eta, 1e-2, 0.99)
            imgui.TextUnformatted("Adam")
            changed, ad_beta1 = imgui.SliderFloat("ad_beta1", ad_beta1, 0, 1.)
            changed, ad_beta2 = imgui.SliderFloat("ad_beta2", ad_beta2, 0, 1.)
            changed, ad_eta = imgui.SliderFloat("ad_eta", ad_eta, 1e-2, 0.99)

            # Triangle
            VH = np.vstack([V.T, np.ones((1, V.shape[0]))])
            T = sm.get_transform()
            ABC = (T @ VH)[:3, :]
            A, B, C = ABC[:, 0], ABC[:, 1], ABC[:, 2]
            DX = np.vstack([B - A, C - A]).T
            elen = [
                np.linalg.norm(A - B),
                np.linalg.norm(A - C),
                np.linalg.norm(B - C),
            ]
            # Objective
            sdf = pbat.geometry.sdf.Composite(forest)

            def f_bar(x: np.ndarray) -> float:
                return sdf.eval(DX @ x + A)

            def g_bar(x: np.ndarray) -> np.ndarray:
                h = 1e-4
                gx = sdf.grad(DX @ x + A, h)
                return DX.T @ gx
            
            def f(x: np.ndarray) -> float:
                return sdf.eval(x)

            def g(x: np.ndarray) -> np.ndarray:
                h = 1e-4
                gx = sdf.grad(x, h)
                return gx

            # Controls
            if imgui.Button("Step"):
                xkp1, fkp1, gkp1, Bkp1, Rkp1 = step_minimize_triangle(
                    f_bar,
                    g_bar,
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

                gd_xkp1 = step_proj_gradient_decent(g_bar,gd_xk, gd_eta)
                fw_xkp1 = step_frank_wolfe(g,fw_xk, ABC,len(fw_xpath))

                ad_xkp1, ad_mtp1, ad_vtp1 = step_adam(
                    g_bar,
                    ad_xk,
                    ad_mt,
                    ad_vt,
                    ad_beta1,
                    ad_beta2,
                    ad_eta,
                    len(ad_xpath)
                )

                if np.linalg.norm(xk - xkp1) > 0.0:
                    xpath = xpath + [DX @ xkp1 + A]
                    fpath = fpath + [fkp1]
                    Rpath = Rpath + [Rkp1]
                xk, fk, gk, Bk, Rk = xkp1, fkp1, gkp1, Bkp1, Rkp1
         
                if np.linalg.norm(gd_xk - gd_xkp1) > 0.0:
                    gd_xpath = gd_xpath + [DX @ gd_xkp1 + A]
                    gd_fpath = gd_fpath + [f_bar(gd_xkp1)]
                gd_xk = gd_xkp1

                if np.linalg.norm(fw_xk - fw_xkp1) > 0.0:
                    fw_xpath = fw_xpath + [fw_xkp1]
                    fw_fpath = fw_fpath + [f(fw_xkp1)]
                fw_xk = fw_xkp1

                if np.linalg.norm(ad_xk - ad_xkp1) > 0.0:
                    ad_xpath = ad_xpath + [DX @ ad_xkp1 + A]
                    ad_fpath = ad_fpath + [f_bar(ad_xkp1)]
                ad_xk, ad_mt, ad_vt = ad_xkp1, ad_mtp1, ad_vtp1
                
            changed, randomize_sample = imgui.Checkbox("Randomize sample", randomize_sample)
            if imgui.Button("Reset" if len(xpath) > 0 else "Start"):
                if randomize_sample:
                    xk = sample_in_reference_triangle_2d()
                else:
                    xk = np.array([0.25, 0.25])
                gd_xk = xk.copy()
                fw_xk = DX @ xk.copy() + A
                ad_xk = xk.copy()

                fk = f_bar(xk)
                gk = g_bar(xk)

                Bk = np.eye(2) * sigmaB * max(elen)
                Rk = sigmaR * max(elen)
                xpath = [DX @ xk + A]
                fpath = [fk]
                Rpath = [Rk]

                gd_fk = f_bar(gd_xk)
                gd_xpath = [DX @ gd_xk + A]
                gd_fpath = [gd_fk]

                fw_fk = f(fw_xk)
                fw_xpath = [fw_xk]
                fw_fpath = [fw_fk]

                ad_fk = f_bar(ad_xk)
                ad_xpath = [DX @ ad_xk + A]
                ad_fpath = [ad_fk]

            if len(xpath) > 0:
                VE = np.array(xpath)
                EE = np.vstack([np.arange(len(xpath) - 1), np.arange(1, len(xpath))]).T
                cn = ps.register_curve_network(
                    "Optimization Path",
                    VE,
                    EE,
                )
                pc = ps.register_point_cloud("SR1 xk", VE[-1:, :])
                cn.set_ignore_slice_plane(slice_plane, True)
                pc.set_ignore_slice_plane(slice_plane, True)
                pc.set_radius(1.1 * cn.get_radius(), relative=False)

            if len(gd_xpath) > 0:
                VE = np.array(gd_xpath)
                EE = np.vstack([np.arange(len(gd_xpath) - 1), np.arange(1, len(gd_xpath))]).T
                cn = ps.register_curve_network(
                    "Optimization Path GD",
                    VE,
                    EE,
                )
                pc = ps.register_point_cloud("GD xk", VE[-1:, :])
                cn.set_ignore_slice_plane(slice_plane, True)
                pc.set_ignore_slice_plane(slice_plane, True)
                pc.set_radius(1.1 * cn.get_radius(), relative=False)

            if len(fw_xpath) > 0:
                VE = np.array(fw_xpath)
                EE = np.vstack([np.arange(len(fw_xpath) - 1), np.arange(1, len(fw_xpath))]).T
                cn = ps.register_curve_network(
                    "Optimization Path FW",
                    VE,
                    EE,
                )
                pc = ps.register_point_cloud("FW xk", VE[-1:, :])
                cn.set_ignore_slice_plane(slice_plane, True)
                pc.set_ignore_slice_plane(slice_plane, True)
                pc.set_radius(1.1 * cn.get_radius(), relative=False)

            if len(ad_xpath) > 0:
                VE = np.array(ad_xpath)
                EE = np.vstack([np.arange(len(ad_xpath) - 1), np.arange(1, len(ad_xpath))]).T
                cn = ps.register_curve_network(
                    "Optimization Path Adam",
                    VE,
                    EE,
                )
                pc = ps.register_point_cloud("Ad xk", VE[-1:, :])
                cn.set_ignore_slice_plane(slice_plane, True)
                pc.set_ignore_slice_plane(slice_plane, True)
                pc.set_radius(1.1 * cn.get_radius(), relative=False)

            if len(fpath) > 0:
                if implot.BeginPlot("Signed distance"):
                    implot.PlotLine(
                        "sdf", np.array(fpath), 1 / len(fpath), 0.0  # xscale  # xstart
                    )
                    implot.PlotLine(
                        "GD sdf", np.array(gd_fpath), 1 / len(gd_fpath), 0.0
                    )
                    implot.PlotLine(
                        "FW sdf", np.array(fw_fpath), 1 / len(fw_fpath), 0.0
                    )
                    implot.PlotLine(
                        "AD sdf", np.array(ad_fpath), 1 / len(ad_fpath), 0.0
                    )
                    implot.EndPlot()
            if len(Rpath) > 0:
                if implot.BeginPlot("Trust Region Radius"):
                    implot.PlotLine(
                        "Rk",
                        np.array(Rpath) / max(elen),
                        1 / len(Rpath),
                        0.0,  # xscale  # xstart
                    )
                    implot.EndPlot()

            imgui.TreePop()

    ps.set_user_callback(callback)
    ps.show()
