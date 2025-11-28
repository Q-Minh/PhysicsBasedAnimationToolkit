# type: ignore
from pbatoolkit import pbat
import polyscope as ps
import polyscope.imgui as imgui
import polyscope.implot as implot
from .solvers.base import BaseSolver
import numpy as np


plot_flags = [
    # ("NoLabel", implot.ImPlotAxisFlags_NoLabel),
    ("NoGridLines", implot.ImPlotAxisFlags_NoGridLines),
    ("NoTickMarks", implot.ImPlotAxisFlags_NoTickMarks),
    ("NoTickLabels", implot.ImPlotAxisFlags_NoTickLabels),
    # ("NoInitialFit", implot.ImPlotAxisFlags_NoInitialFit),
    # ("NoMenus", implot.ImPlotAxisFlags_NoMenus),
    # ("NoSideSwitch", implot.ImPlotAxisFlags_NoSideSwitch),
    # ("NoHighlight", implot.ImPlotAxisFlags_NoHighlight),
    # ("Opposite", implot.ImPlotAxisFlags_Opposite),
    # ("Foreground", implot.ImPlotAxisFlags_Foreground),
    # ("Invert", implot.ImPlotAxisFlags_Invert),
    ("AutoFit", implot.ImPlotAxisFlags_AutoFit),
    # ("RangeFit", implot.ImPlotAxisFlags_RangeFit),
    # ("PanStretch", implot.ImPlotAxisFlags_PanStretch),
    # ("LockMin", implot.ImPlotAxisFlags_LockMin),
    # ("LockMax", implot.ImPlotAxisFlags_LockMax),
    # ("Lock", implot.ImPlotAxisFlags_Lock),
    # ("NoDecorations", implot.ImPlotAxisFlags_NoDecorations),
    # ("AuxDefault", implot.ImPlotAxisFlags_AuxDefault),
]


def reduce_flags(selected_flags: list[bool]) -> int:
    flags = implot.ImPlotAxisFlags_None
    for i, selected in enumerate(selected_flags):
        if selected:
            flags |= plot_flags[i][1]
    return flags


class Convergence:
    _f: list[list[float]]
    _gnorm: list[list[float]]
    _solver_names: list[str]
    _convergence_analysis_requested: bool
    _plot_flag_mask: list[bool]

    def __init__(self):
        self._f = []
        self._gnorm = []
        self._solver_names = []
        self._convergence_analysis_requested = False
        self._plot_flag_mask = [False for _ in plot_flags]

    def draw(self):
        default_button_size = [imgui.GetWindowWidth() / 2.1, 0]
        imgui.PushID("Convergence")
        if imgui.TreeNode("Plot Options"):
            for i, (flag_name, flag_value) in enumerate(plot_flags):
                _, self._plot_flag_mask[i] = imgui.Checkbox(
                    flag_name,
                    self._plot_flag_mask[i],
                )
            imgui.TreePop()
        if imgui.Button("Step", default_button_size):
            self._convergence_analysis_requested = True
        flags = reduce_flags(self._plot_flag_mask)
        if implot.BeginPlot("Objective"):
            implot.SetupAxes(
                "Iteration",
                "f",
                flags,
                flags,
            )
            for solver_name, f_vals in zip(self._solver_names, self._f):
                implot.PlotLine(
                    solver_name,
                    np.arange(len(f_vals)),
                    np.array(f_vals),
                )
            implot.EndPlot()
        if implot.BeginPlot("Gradient"):
            implot.SetupAxes(
                "Iteration",
                "||g||",
                flags,
                flags,
            )
            for solver_name, gnorm_vals in zip(self._solver_names, self._gnorm):
                implot.PlotLine(
                    solver_name,
                    np.arange(len(gnorm_vals)),
                    np.array(gnorm_vals),
                )
            implot.EndPlot()
        imgui.PopID()

    def analyze_convergence(
        self,
        selected: int,
        solvers: list[BaseSolver],
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        self._solver_names = [solver.name for solver in solvers]
        self._f = [[] for _ in solvers]
        self._gnorm = [[] for _ in solvers]
        x0 = fem.x.copy()
        v0 = fem.v.copy()
        xstar = np.zeros_like(fem.x)
        vstar = np.zeros_like(fem.v)
        for s, solver in enumerate(solvers):
            fem.x = x0
            fem.v = v0
            solver.solve(
                fem,
                contact,
                lambda: self.collect_iteration_data(s, fem, contact),
            )
            if s == selected:
                xstar = fem.x
                vstar = fem.v
        fem.x = xstar
        fem.v = vstar
        self._convergence_analysis_requested = False

    def collect_iteration_data(
        self,
        s: int,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        fs = fem.objective(fem.x)
        gs = fem.gradient(fem.x)
        gsnorm = np.linalg.norm(gs)
        self._f[s].append(fs)
        self._gnorm[s].append(gsnorm)

    @property
    def is_convergence_analysis_requested(self) -> bool:
        return self._convergence_analysis_requested
