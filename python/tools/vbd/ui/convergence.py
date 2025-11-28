# type: ignore
from pbatoolkit import pbat
import polyscope as ps
import polyscope.imgui as imgui
import polyscope.implot as implot
from .solvers.base import BaseSolver
import numpy as np


plot_flags = [
    ("NoLabel", implot.ImPlotAxisFlags_NoLabel),
    ("NoGridLines", implot.ImPlotAxisFlags_NoGridLines),
    ("NoTickMarks", implot.ImPlotAxisFlags_NoTickMarks),
    ("NoTickLabels", implot.ImPlotAxisFlags_NoTickLabels),
    ("NoInitialFit", implot.ImPlotAxisFlags_NoInitialFit),
    ("NoMenus", implot.ImPlotAxisFlags_NoMenus),
    ("NoSideSwitch", implot.ImPlotAxisFlags_NoSideSwitch),
    ("NoHighlight", implot.ImPlotAxisFlags_NoHighlight),
    ("Opposite", implot.ImPlotAxisFlags_Opposite),
    ("Foreground", implot.ImPlotAxisFlags_Foreground),
    ("Invert", implot.ImPlotAxisFlags_Invert),
    ("AutoFit", implot.ImPlotAxisFlags_AutoFit),
    ("RangeFit", implot.ImPlotAxisFlags_RangeFit),
    ("PanStretch", implot.ImPlotAxisFlags_PanStretch),
    ("LockMin", implot.ImPlotAxisFlags_LockMin),
    ("LockMax", implot.ImPlotAxisFlags_LockMax),
    ("Lock", implot.ImPlotAxisFlags_Lock),
    ("NoDecorations", implot.ImPlotAxisFlags_NoDecorations),
    ("AuxDefault", implot.ImPlotAxisFlags_AuxDefault),
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

    def step(self):
        # TODO: Step all solvers and collect convergence data for real
        self._f = [[1, 2, 3], [-1, 0, 1, 2], [3, 4, 5]]
        self._gnorm = [np.arange(5) for _ in range(3)]
        self._solver_names = ["Solver A", "Solver B", "Solver C"]
        self._convergence_analysis_requested = False

    @property
    def should_step(self) -> bool:
        return self._convergence_analysis_requested
