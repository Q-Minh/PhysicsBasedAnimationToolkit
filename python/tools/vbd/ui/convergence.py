# type: ignore
from pbatoolkit import pbat
import polyscope as ps
import polyscope.imgui as imgui
import polyscope.implot as implot
from .solvers.base import BaseSolver
import numpy as np


class Convergence:
    _f: list[list[float]]
    _gnorm2: list[list[float]]
    _errors: list[list[float]]
    _solver_names: list[str]
    _convergence_analysis_requested: bool

    def __init__(self):
        self._f = []
        self._gnorm2 = []
        self._errors = []
        self._solver_names = []
        self._convergence_analysis_requested = False

    def draw(self):
        default_button_size = [imgui.GetWindowWidth() / 2.1, 0]
        imgui.PushID("Convergence")
        imgui.Text(
            "Note that contact detection will be computed every iteration\n"
            "no matter the solver and contact parameters."
        )
        if imgui.Button("Step", default_button_size):
            self._convergence_analysis_requested = True
        flags = implot.ImPlotAxisFlags_None
        if imgui.Button("Fit Axes", default_button_size):
            flags = flags + implot.ImPlotAxisFlags_AutoFit
        if implot.BeginPlot("Objective"):
            implot.SetupAxes(
                "Iteration",
                "f / max(f)",
                flags,
                flags,
            )
            implot.SetupLegend(implot.ImPlotLocation_NorthEast)
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
                "||g||^2 / max(||g*||^2)",
                flags,
                flags,
            )
            implot.SetupLegend(implot.ImPlotLocation_NorthEast)
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
            implot.SetupLegend(implot.ImPlotLocation_NorthEast)
            for solver_name, error_vals in zip(self._solver_names, self._errors):
                implot.PlotLine(
                    solver_name,
                    np.arange(len(error_vals)),
                    np.array(error_vals),
                )
            implot.EndPlot()
        imgui.PopID()

    def objective(
        self,
        x: np.ndarray,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        return fem.objective(x) + contact.potential(x)

    def gradient(
        self,
        x: np.ndarray,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        gc = contact.gradient(x, for_augmented_lagrangian=False)
        gd = fem.gradient(x)
        return gd + gc

    def analyze_convergence(
        self,
        selected: int,
        solvers: list[BaseSolver],
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        profiler=None
    ):
        self._solver_names = [solver.name for solver in solvers]
        x0 = fem.x.copy()
        v0 = fem.v.copy()
        self._f = [[] for _ in solvers]
        self._gnorm2 = [[] for _ in solvers]
        self._errors = [[] for _ in solvers]
        xs = [[] for _ in solvers]
        xstar = np.zeros_like(fem.x)
        vstar = np.zeros_like(fem.v)
        for s, solver in enumerate(solvers):
            if profiler is not None:
                profiler.begin_frame("Convergence")
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
            if profiler is not None:
                profiler.end_frame("Convergence")
        fem.x = xstar
        fem.v = vstar
        dx = lambda a, b: a.ravel() - b.ravel()
        dxstarnorm2 = np.dot(dx(x0, xstar), dx(x0, xstar))
        zero = 1e-12
        self._errors = [
            [
                (np.dot(dx(x, xstar), dx(x, xstar))) / max(dxstarnorm2, zero)
                for x in xs[s]
            ]
            for s in range(len(solvers))
        ]
        fmax = np.max([np.max(self._f[s]) for s in range(len(solvers))])
        self._f = [[fval / fmax for fval in self._f[s]] for s in range(len(solvers))]
        gmax2 = np.max([np.max(gnorm2) for gnorm2 in self._gnorm2])
        self._gnorm2 = [
            [gnorm2 / gmax2 for gnorm2 in self._gnorm2[s]] for s in range(len(solvers))
        ]
        self._convergence_analysis_requested = False

    def collect_iteration_data(
        self,
        s: int,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        xs: list[list[np.ndarray]],
    ):
        xt = -fem.bdf.inertia().reshape((3, -1), order="F")
        contact.linearize_constraints(fem.x, xt)
        fs = self.objective(fem.x, fem, contact)
        gs = self.gradient(fem.x, fem, contact)
        gsnorm2 = np.dot(gs, gs)
        self._f[s].append(fs)
        self._gnorm2[s].append(gsnorm2)
        xs[s].append(fem.x.copy())

    @property
    def is_convergence_analysis_requested(self) -> bool:
        return self._convergence_analysis_requested
