"""The CPU editor's display of material parameters: names, scales and colormaps, shared by the GUIs that show a
mesh's materials. Needs numpy only."""

import numpy as np

# The quantities' names, as the CPU editor shows them
LAME_MU, LAME_LAMBDA, YOUNG_MODULUS, MASS = "Lame mu", "Lame lambda", "Young's Modulus", "Mass"
FEM_QUANTITIES = (LAME_MU, LAME_LAMBDA, YOUNG_MODULUS, MASS)
# The lumped masses' color range: these percentiles of the masses, so that a few heavy nodes don't wash out the rest
MASS_PERCENTILES = (2.0, 98.0)


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
    """The editor's display scale of material parameters: log10(value + 1)."""
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


def mass_display(m: np.ndarray, log10: bool = True) -> tuple[np.ndarray, tuple[float, float]]:
    """Lumped masses' displayed values and color range.

    Sub-kilogram masses are near 0 on the log10(value + 1) scale, and a few heavy nodes would set the top of a
    min-max range, so the masses show as log10(m) (or linear) over their ``MASS_PERCENTILES``.

    Args:
        m (np.ndarray): (N,) masses (kg). Massless nodes (e.g. static colliders') show at the smallest positive
            mass.
        log10 (bool): log10(m), or m.

    Returns:
        tuple[np.ndarray, tuple[float, float]]: (N,) values to show and their color range.
    """
    m = np.asarray(m, dtype=np.float64)
    positive = m[m > 0] if np.any(m > 0) else np.ones(1)
    shown = np.maximum(m, positive.min())
    if log10:
        shown, positive = np.log10(shown), np.log10(positive)
    lo, hi = (float(v) for v in np.percentile(positive, MASS_PERCENTILES))
    return shown, (lo, hi if hi > lo else lo + max(abs(lo), 1.0) * 1e-6)


def display(name: str, values: np.ndarray, log10: bool = True) -> tuple[np.ndarray, dict]:
    """A ``FEM_QUANTITIES`` quantity's displayed values and Polyscope ``add_scalar_quantity`` color options.

    Args:
        name (str): One of ``FEM_QUANTITIES``.
        values (np.ndarray): Its values (``fem_parameters``).
        log10 (bool): Material parameters on the editor's log10(value + 1) scale, with its fixed range for
            Young's modulus, and masses in log10(m) (``mass_display``); or linear, over the values' range
            (masses: over their percentiles).

    Returns:
        tuple[np.ndarray, dict]: The values to show, and their ``cmap`` (and ``vminmax`` for masses, and for
        Young's modulus on the log scale).
    """
    cmap = mass_density_log10_cmap() if name == MASS else (
        young_modulus_log10_cmap() if name == YOUNG_MODULUS else lame_parameters_cmap())
    if name == MASS:
        shown, vminmax = mass_display(values, log10)
        return shown, dict(cmap=cmap, vminmax=vminmax)
    if not log10:
        return np.asarray(values, dtype=np.float64), dict(cmap=cmap)
    options = dict(cmap=cmap)
    if name == YOUNG_MODULUS:
        options["vminmax"] = young_modulus_log10_range()
    return log10_scale(values), options


def fem_quantities(lamegU: np.ndarray, egU: np.ndarray, n_elements: int,
                   m: np.ndarray) -> list[tuple[str, np.ndarray, str, dict]]:
    """A volume mesh's material and mass quantities, as the CPU editor shows them (masses: ``mass_display``).

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


import unittest


class TestMaterial(unittest.TestCase):
    def test_mass_range(self):
        """Masses show as log10(m) over their 2-98th percentiles: a few heavy nodes don't set the top of the
        range, and massless nodes show at the smallest positive mass."""
        m = np.concatenate([np.linspace(0.01, 0.05, 96), [0.4, 0.5, 0.0, 0.0]])
        shown, options = display(MASS, m)
        positive = m[m > 0]
        np.testing.assert_allclose(options["vminmax"], np.percentile(np.log10(positive), MASS_PERCENTILES))
        self.assertLess(options["vminmax"][1], np.log10(0.1))  # the heavy outliers are above the range
        np.testing.assert_allclose(shown[:98], np.log10(m[:98]))
        np.testing.assert_allclose(shown[98:], np.log10(0.01))
        self.assertEqual(options["cmap"], mass_density_log10_cmap())
        linear, options = display(MASS, m, log10=False)
        np.testing.assert_allclose(options["vminmax"], np.percentile(positive, MASS_PERCENTILES))
        np.testing.assert_allclose(linear[98:], 0.01)
        # Uniform or massless meshes still get a nonempty range
        for m in (np.full(5, 0.2), np.zeros(5)):
            lo, hi = display(MASS, m)[1]["vminmax"]
            self.assertTrue(np.isfinite([lo, hi]).all() and hi > lo)

    def test_materials_unchanged(self):
        """Young's modulus keeps the editor's fixed log10(value + 1) range, the Lame parameters their data range."""
        values, options = display(YOUNG_MODULUS, np.array([1e6, 1e8]))
        np.testing.assert_allclose(values, np.log10([1e6 + 1, 1e8 + 1]))
        self.assertEqual(options["vminmax"], young_modulus_log10_range())
        self.assertNotIn("vminmax", display(LAME_MU, np.array([1e5, 1e7]))[1])


if __name__ == "__main__":
    unittest.main()
