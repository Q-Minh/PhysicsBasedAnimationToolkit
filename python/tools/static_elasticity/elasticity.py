from pbatoolkit import pbat, pypbat
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import time
import meshio
import argparse
import math
import gc
import scipy as sp


def signal(w: float, v: np.ndarray, t: float, c: float, k: float):
    u = c*np.sin(k*w*t)*v
    return u

def parse_archive_path(spec: str) -> tuple[str, str]:
    """Parse 'file.h5:group/path' into (file_path, group_path)."""
    parts = spec.split(":")
    if len(parts) >= 2:
        return parts[0], ":".join(parts[1:])
    return parts[0], ""

def load_fem_dynamics(spec: str) -> pbat.sim.dynamics.FemElastoDynamics:
    """Deserialize FemElastoDynamics from an h5 archive at file:group."""
    file_path, group_path = parse_archive_path(spec)
    archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.ReadOnly)
    fem_cpu = pbat.sim.dynamics.FemElastoDynamics()
    fem_cpu.deserialize(archive[group_path] if group_path else "/")
    archive = None
    gc.collect()
    return fem_cpu


def hyper_elastic_modes(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    modes: int = 30,
    sigma: float = -1e-5,
    zero: float = 0.0,
):
    """Computes natural (linear) displacement modes of mesh at current configuration."""
    E = fem.E
    X = fem.X
    x = np.ravel(fem.x, order="F")
    wgU = fem.wgU
    egU = fem.egU
    GNegU = fem.GNegU
    n_nodes=X.shape[1]
    _, _, HU = pbat.fem.hyper_elastic_potential(
        E=E,
        n_nodes=n_nodes,
        eg=np.ravel(egU),
        wg=np.ravel(wgU),
        GNeg=GNegU,
        mug=np.ravel(fem.lamegU[0, :].copy()),
        lambdag=np.ravel(fem.lamegU[1, :].copy()),
        x=x,
        energy=pbat.fem.HyperElasticEnergy.StableNeoHookean,
        flags=pbat.fem.ElementElasticityComputationFlags.Hessian,
        spd_correction=pbat.fem.HyperElasticSpdCorrection.NoCorrection,
        element=pbat.fem.Element.Tetrahedron,
        order=1,
        dims=X.shape[0],
    )
    n = 3*fem.X.shape[1]
    n_red = 3*(n_nodes-fem.ndbc)
    free_nodes = fem.dbc[0:n_nodes-fem.ndbc]
    free_dofs = np.concatenate([3*free_nodes, 3*free_nodes+1, 3*free_nodes+2])
    m = fem.M()[free_dofs]
    M_red = sp.sparse.diags_array([m], offsets=[0], shape=(n_red, n_red), format="dia")
    HU_red = HU[:, free_dofs].tocsr()[free_dofs, :]
    modes = min(modes, n)
    
    l, V_red = sp.sparse.linalg.eigsh(HU_red, k=modes, M=M_red, sigma=sigma, which="LM")
    V_red = V_red / sp.linalg.norm(V_red, axis=0, keepdims=True)
    V = np.zeros((n, modes))
    V[free_dofs, :] = V_red
    l[l <= zero] = 0
    w = np.sqrt(l)
    return w, V

class ModeParams:
    def __init__(self, mode: int, amplitude: float):
        self.mode = mode
        self.amplitude = amplitude
        self.max_amplitude = 100.0
        self.min_amplitude = 0.0
        self.show_more = False
    
    def draw(self):
        imgui.PushID(self.mode)
        imgui.PushItemWidth(imgui.GetWindowWidth() * 0.5)
        imgui.Text(f"Mode {self.mode:03}")
        imgui.SameLine()
        _, self.amplitude  = imgui.SliderFloat("Amplitude", self.amplitude, self.min_amplitude, self.max_amplitude)
        imgui.SameLine()
        label = "More" if not self.show_more else "Less"
        if imgui.Button(label):
            self.show_more = not self.show_more
        imgui.PopItemWidth()
        if self.show_more:
            imgui.PushItemWidth(imgui.GetWindowWidth() * 0.3)
            imgui.Text("")
            imgui.SameLine()
            _, self.max_amplitude = imgui.SliderFloat("Max Amplitude", self.max_amplitude, 0.0, 1000.0)
            imgui.SameLine()
            _, self.min_amplitude = imgui.SliderFloat("Min Amplitude", self.min_amplitude, 0.0, 1000.0)
            imgui.PopItemWidth()

        #self.frequency = imgui.SliderFloat("Frequency", self.frequency, 0.0, 1.0)
        
        #self.frequency = imgui.SliderFloat("Frequency", self.frequency, 0, 1)
        imgui.PopID()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Elastic stiffness modal decomposition demo",
    )
    parser.add_argument("-i", "--input", help="file.h5:group/path to a serialized FemElastoDynamics object.", type=str,
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

    zero = 0.0
    sigma = -1e-5
    fem_loaded = load_fem_dynamics(args.input)
    w, Veigs = hyper_elastic_modes(
        fem=fem_loaded,
        modes=args.modes,
        sigma=sigma,
        zero=zero,
    )
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_volume_mesh("model", fem_loaded.X.T, fem_loaded.E.T)
    mode = 6
    t0 = time.time()
    t = 0
    c = 0
    k = 1
    mode_params = [ModeParams(i, c) for i in range(args.modes)]

    def callback():
        global mode, c, k, args
        V = fem_loaded.X.T
        for param in mode_params:
            param.draw()
            u = param.amplitude*np.sin(w[param.mode])*Veigs[:, param.mode]
            V += u.reshape(fem_loaded.X.shape[1], 3)

        # t = time.time() - t0
        # V = fem_loaded.X.T + signal(w[mode], Veigs[:, mode],
        #                       t, c, k).reshape(fem_loaded.X.shape[1], 3)
        vm.update_vertex_positions(V)

    ps.set_user_callback(callback)
    ps.show()
