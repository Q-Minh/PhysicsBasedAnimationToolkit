import gpytoolbox as gpt
import numpy as np
import meshio
import argparse


def beam(
    dims: np.ndarray | list[float],
    resolution: np.ndarray | list[int],
    subdivision: str = "5",
    center: bool = False,
    normalize: bool = False,
):
    """
    Generate a beam mesh.

    Parameters
    ----------
    dims : list of float
        Dimensions of the beam in the format: dx dy dz
    resolution : list of int
        Mesh resolution for the beam in the format: nx ny nz
    center : bool
        Center the mesh at the origin.
    normalize : bool
        Rescale the mesh so that the largest dimension is unit.

    Returns
    -------
    Tuple[np.ndarray[float], np.ndarray[int]] : The generated beam mesh (V,T).
    """
    V, T = gpt.regular_cube_mesh(
        resolution[0],
        resolution[1],
        resolution[2],
        type=(
            "rotationally-symmetric"
            if subdivision == "5"
            else "reflectionally-symmetric"
        ),
    )
    # Scale the mesh to the desired dimensions
    V *= np.array(dims)
    # Center the mesh at the origin if requested
    if center:
        V -= np.array(dims) / 2
    # Normalize the mesh if requested
    if normalize:
        max_dim = max(dims)
        V /= max_dim
    return V, T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a beam mesh.")
    parser.add_argument(
        "--dims",
        nargs="+",
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Dimensions of the beam in the format: dx dy dz",
    )
    parser.add_argument(
        "--resolution",
        nargs="+",
        type=int,
        default=[40, 10, 10],
        help="Mesh resolution for the beam in the format: nx ny nz",
    )
    parser.add_argument(
        "--subdivision",
        type=str,
        default="5",
        help="5 | 6 tets per cube (default: 5).",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center the mesh at the origin.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Rescale the mesh so that the largest dimension is unit.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="beam.mesh",
        help="Output filename for the mesh.",
    )
    args = parser.parse_args()
    V, T = beam(
        dims=args.dims,
        resolution=args.resolution,
        subdivision=args.subdivision,
        center=args.center,
        normalize=args.normalize,
    )
    meshio.write(args.output, meshio.Mesh(V, [("tetra", T)]))
