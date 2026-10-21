"""The CPU editor's display of material parameters: names, scales and colormaps, shared by the GUIs that show a
mesh's materials. Needs numpy only."""

import numpy as np

# The quantities' names, as the CPU editor shows them
LAME_MU, LAME_LAMBDA, YOUNG_MODULUS, MASS = "Lame mu", "Lame lambda", "Young's Modulus", "Mass"
FEM_QUANTITIES = (LAME_MU, LAME_LAMBDA, YOUNG_MODULUS, MASS)


def young_modulus_log10_range() -> tuple[float, float]:
    return (5.0, 9.0)

def young_modulus_log10_cmap() -> str:
    return "blues"

def poisson_ratio_range() -> tuple[float, float]:
    return (0.0, 0.5)

def poisson_ratio_cmap() -> str:
    return "blues"

def mass_density_log10_range() -> tuple[float, float]:
    return (1.0, 6.0)

def mass_density_log10_cmap() -> str:
    return "reds"

def lame_parameters_cmap() -> str:
    return "blues"

def regions_cmap() -> str:
    return "viridis"


def log10_scale(values: np.ndarray) -> np.ndarray:
    """The editor's display scale of material parameters and masses: log10(value + 1)."""
    return np.log10(np.asarray(values, dtype=np.float64) + 1)


def element_lame_parameters(lamegU: np.ndarray, egU: np.ndarray, n_elements: int) -> tuple[np.ndarray, np.ndarray]:
    """Each element's Lame parameters mu and lambda (Pa): the mean over its quadrature points.

    Args:
        lamegU (np.ndarray): (2, Q) mu and lambda per quadrature point (a FemElastoDynamics' ``lamegU``).
        egU (np.ndarray): (Q,) or (Q, 1) the element of each quadrature point.
        n_elements (int): M.

    Returns:
        tuple[np.ndarray, np.ndarray]: (M,) mu, (M,) lambda.
    """
    e = np.asarray(egU, dtype=np.int64).ravel()
    count = np.maximum(np.bincount(e, minlength=n_elements), 1)
    mu = np.bincount(e, np.asarray(lamegU[0], dtype=np.float64), n_elements) / count
    lam = np.bincount(e, np.asarray(lamegU[1], dtype=np.float64), n_elements) / count
    return mu, lam


def young_modulus(mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Young's modulus mu (3 lambda + 2 mu) / (lambda + mu) of Lame parameters."""
    return mu * (3 * lam + 2 * mu) / np.maximum(lam + mu, 1e-30)


def fem_parameters(lamegU: np.ndarray, egU: np.ndarray, n_elements: int, m: np.ndarray) -> dict[str, np.ndarray]:
    """A FEM mesh's material parameters per element and lumped masses per node.

    Args:
        lamegU (np.ndarray): (2, Q) mu and lambda per quadrature point.
        egU (np.ndarray): (Q,) the element of each quadrature point.
        n_elements (int): M.
        m (np.ndarray): (N,) lumped mass per node (a FemElastoDynamics' ``m``; kg).

    Returns:
        dict[str, np.ndarray]: ``FEM_QUANTITIES`` -> (M,) mu, lambda and Young's modulus (Pa), and (N,) masses.
    """
    mu, lam = element_lame_parameters(lamegU, egU, n_elements)
    return dict(zip(FEM_QUANTITIES, (mu, lam, young_modulus(mu, lam), np.asarray(m, dtype=np.float64).ravel())))


def is_nodal(name: str) -> bool:
    """Whether a ``FEM_QUANTITIES`` quantity is per node (the mass) rather than per element."""
    return name == MASS


def display(name: str, values: np.ndarray, log10: bool = True) -> tuple[np.ndarray, dict]:
    """A ``FEM_QUANTITIES`` quantity's displayed values and Polyscope ``add_scalar_quantity`` color options.

    Args:
        name (str): One of ``FEM_QUANTITIES``.
        values (np.ndarray): Its values (``fem_parameters``).
        log10 (bool): On the editor's log10(value + 1) scale, with its fixed range for Young's modulus, or
            linear over the values' range.

    Returns:
        tuple[np.ndarray, dict]: The values to show, and their ``cmap`` (and ``vminmax`` for Young's modulus on
        the log scale).
    """
    cmap = mass_density_log10_cmap() if name == MASS else (
        young_modulus_log10_cmap() if name == YOUNG_MODULUS else lame_parameters_cmap())
    if not log10:
        return np.asarray(values, dtype=np.float64), dict(cmap=cmap)
    options = dict(cmap=cmap)
    if name == YOUNG_MODULUS:
        options["vminmax"] = young_modulus_log10_range()
    return log10_scale(values), options


def fem_quantities(lamegU: np.ndarray, egU: np.ndarray, n_elements: int,
                   m: np.ndarray) -> list[tuple[str, np.ndarray, str, dict]]:
    """A volume mesh's material and mass quantities, as the CPU editor shows them.

    Args:
        lamegU, egU, n_elements, m: As in ``fem_parameters``.

    Returns:
        list[tuple[str, np.ndarray, str, dict]]: (name, values, ``defined_on`` ("cells" or "vertices"), color
        options) per ``FEM_QUANTITIES`` quantity, for ``add_scalar_quantity(name, values, defined_on=...,
        **options)``.
    """
    quantities = []
    for name, values in fem_parameters(lamegU, egU, n_elements, m).items():
        shown, options = display(name, values)
        quantities.append((name, shown, "vertices" if is_nodal(name) else "cells", options))
    return quantities
