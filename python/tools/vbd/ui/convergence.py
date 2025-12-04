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
    _gnorm2: list[list[float]]
    _errors: list[list[float]]
    _solver_names: list[str]
    _convergence_analysis_requested: bool
    _plot_flag_mask: list[bool]

    def __init__(self):
        self._f = []
        self._gnorm2 = []
        self._errors = []
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
                "f / f*",
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
                "||g||^2 / ||g*||^2",
                flags,
                flags,
            )
            for solver_name, gnorm_vals in zip(self._solver_names, self._gnorm2):
                implot.PlotLine(
                    solver_name,
                    np.arange(len(gnorm_vals)),
                    np.array(gnorm_vals),
                )
            implot.EndPlot()
        if implot.BeginPlot("Error"):
            implot.SetupAxes(
                "Iteration",
                "||x - x*||^2 / ||x0 - x*||^2",
                flags,
                flags,
            )
            for solver_name, error_vals in zip(self._solver_names, self._errors):
                implot.PlotLine(
                    solver_name,
                    np.arange(len(error_vals)),
                    np.array(error_vals),
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
        x0 = fem.x.copy()
        v0 = fem.v.copy()
        f0 = fem.objective(x0)
        g0 = fem.gradient(x0)
        self._f = [[f0] for _ in solvers]
        self._gnorm2 = [[np.dot(g0, g0)] for _ in solvers]
        self._errors = [[] for _ in solvers]
        xs = [[x0.copy()] for _ in solvers]
        xstar = np.zeros_like(fem.x)
        vstar = np.zeros_like(fem.v)
        for s, solver in enumerate(solvers):
            fem.x = x0
            fem.v = v0
            solver.solve(
                fem,
                contact,
                lambda: self.collect_iteration_data(s, fem, contact, xs),
            )
            if s == selected:
                xstar = fem.x.copy()
                vstar = fem.v.copy()
        fem.x = xstar
        fem.v = vstar
        dx = lambda a, b: a.ravel() - b.ravel()
        dxstarnorm2 = np.dot(dx(x0, xstar), dx(x0, xstar))
        self._errors = [
            [
                (np.dot(dx(x, xstar), dx(x, xstar)) + 1) / (dxstarnorm2 + 1)
                for x in xs[s]
            ]
            for s in range(len(solvers))
        ]
        fstar = fem.objective(fem.x)
        gstar = fem.gradient(fem.x)
        gstarnorm2 = np.dot(gstar, gstar)
        self._f = [
            [(fval + 1) / (fstar + 1) for fval in self._f[s]]
            for s in range(len(solvers))
        ]
        self._gnorm2 = [
            [(gnorm2 + 1) / (gstarnorm2 + 1) for gnorm2 in self._gnorm2[s]]
            for s in range(len(solvers))
        ]
        self._convergence_analysis_requested = False

    def collect_iteration_data(
        self,
        s: int,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        xs: list[list[np.ndarray]],
    ):
        fs = fem.objective(fem.x)
        gs = fem.gradient(fem.x)
        gsnorm2 = np.dot(gs, gs)
        self._f[s].append(fs)
        self._gnorm2[s].append(gsnorm2)
        xs[s].append(fem.x.copy())

    @property
    def is_convergence_analysis_requested(self) -> bool:
        return self._convergence_analysis_requested
