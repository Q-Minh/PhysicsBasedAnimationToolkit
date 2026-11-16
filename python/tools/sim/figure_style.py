"""
Figure style shared by the paper's figures (``benchmark_plot.py``, ``profiling/paper_figures.py``, ...): the
solvers' fixed colors and display names, and the paper's fonts (the Linux Libertine family of ACM's acmart template).
Only Matplotlib and the standard library, so figure scripts that import it run without h5py, pbatoolkit or Warp.
"""

import glob
import os
import warnings

import matplotlib

# Okabe-Ito colors (colorblind-safe), fixed per solver so a solver keeps its color in every figure
SOLVER_COLORS = {
    "VBD": "#0072B2",
    "AAAVBD": "#D55E00",
    "Chebyshev": "#009E73",
    "AndersonSolver": "#CC79A7",
    "Newton": "#E69F00",
}
SOLVER_NAMES = {"AndersonSolver": "Anderson"}

# The Linux Libertine family of ACM's acmart template (SIGGRAPH, TOG): Linux Libertine (serif, body text), Linux
# Biolinum (sans serif, headings) and Linux Libertine Mono (code). OpenType family names first, then other names.
PAPER_SANS = ("Linux Biolinum O", "Linux Biolinum", "Libertinus Sans")
PAPER_SERIF = ("Linux Libertine O", "Linux Libertine", "Libertinus Serif")
PAPER_MONO = ("Linux Libertine Mono O", "Libertinus Mono")


def register_paper_fonts(font_dir: str | None = None) -> tuple[str | None, str | None, str | None]:
    """Registers the Linux Libertine family with Matplotlib's font manager: the fonts in ``font_dir``, else in the
    ``PBAT_FONT_DIR`` environment variable's directory, else the installed fonts (also ones installed after
    Matplotlib built its font cache). Returns the family names found for Linux Biolinum (the sans serif), Linux
    Libertine (the serif) and Linux Libertine Mono, None for those missing; ``font_manager.findfont`` gives
    their files."""
    from matplotlib import font_manager

    def add(paths):
        for path in paths:
            try:
                font_manager.fontManager.addfont(path)
            except (OSError, RuntimeError, ValueError):
                pass

    for directory in (font_dir, os.environ.get("PBAT_FONT_DIR")):
        if directory:
            add(sorted(glob.glob(os.path.join(directory, "*.otf")) + glob.glob(os.path.join(directory, "*.ttf"))))

    def first(families):
        names = {f.name for f in font_manager.fontManager.ttflist}
        return next((name for name in families if name in names), None)

    if first(PAPER_SANS) is None or first(PAPER_SERIF) is None:
        add(p for p in font_manager.findSystemFonts()
            if any(key in os.path.basename(p).lower() for key in ("libertin", "biolin")))
    return first(PAPER_SANS), first(PAPER_SERIF), first(PAPER_MONO)


def use_paper_fonts(font_dir: str | None = None) -> str | None:
    """Sets Matplotlib's fonts to the Linux Libertine family: figure text and math in Linux Biolinum (the sans
    serif), with Linux Libertine as the serif and Linux Libertine Mono as the monospace font. The fonts come from
    ``font_dir``, else the ``PBAT_FONT_DIR`` environment variable's directory, else the installed fonts
    (:func:`register_paper_fonts`). Returns the sans family used; without it, warns and keeps Matplotlib's fonts."""
    sans, serif, mono = register_paper_fonts(font_dir)
    matplotlib.rcParams["pdf.fonttype"] = 42  # embed the fonts as TrueType/OpenType, editable text in the PDF
    for key, name in (("font.serif", serif), ("font.monospace", mono)):
        if name is not None:
            matplotlib.rcParams[key] = [name, *matplotlib.rcParams[key]]
    if sans is None:
        warnings.warn("Linux Biolinum not found (--font-dir, PBAT_FONT_DIR or an installed font): using Matplotlib's "
                      "default font")
        return None
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [sans, *matplotlib.rcParams["font.sans-serif"]],
                                "mathtext.fontset": "custom", "mathtext.rm": sans, "mathtext.sf": sans,
                                "mathtext.it": f"{sans}:italic", "mathtext.bf": f"{sans}:bold",
                                "mathtext.bfit": f"{sans}:bold"})  # Biolinum has no bold italic
    return sans
