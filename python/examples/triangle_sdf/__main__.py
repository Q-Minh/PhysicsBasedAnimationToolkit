import argparse
from pbatoolkit import pbat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib




from .plot import plot
from .minimizer import Minimizer
from .minimizers.fw import FW
from .minimizers.pgd import PGD
from .args_processor import process_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser("triangle_sdf")
    parser.add_argument("--json", help="path to JSON file containing configuration parameters")
    parser.add_argument("--sdf", help="path to .h5 file", type=str)
    parser.add_argument("-n", "--numSteps", help="number of steps in non-interactive modes. Defaults to 10", type=int, default=10)
    parser.add_argument("-t", "--triangle", help="triangle's vertices positions in 3D. Defaults to [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],   [0.0, 0.5, 0.5]]", default=[
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.0, 0.5, 0.5],
        ])
    parser.add_argument("-s", "--start", help="optimization start point in barycentric coordinates. Defaults to [0.25, 0.25]", type=list[float], default=[0.25,0.25])
    parser.add_argument("--plot", help="Use matplotlib",action="store_true")
    args = parser.parse_args()
    

    params = process_args(args)

    sdf_forest = pbat.geometry.sdf.Forest()

    if params["sdf"]:
        archive = pbat.io.Archive(params["sdf"], pbat.io.AccessMode.ReadOnly)
        sdf_forest.deserialize(archive)
    
    triangle = params["triangle"]
    A, B, C = triangle
    DX = np.vstack([B - A, C - A]).T

    sdf = pbat.geometry.sdf.Composite(sdf_forest)

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

    pgd = PGD()
    pgd.setup(f_bar,g_bar, eta=0.1)
    fw = FW()
    fw.setup(f,g,vertices=triangle.T)
    minimizers: list[Minimizer] = [pgd, fw]

    if not params["interactive"]:
        positions: dict[str,list] = {}
        evaluations: dict[str,list] = {}
        for minimizer in minimizers:
            positions[minimizer.label] = [params["start"] if minimizer.barycentric else DX @ params["start"] + A]
            evaluations[minimizer.label] = [f_bar(params["start"])]
        
        for i in range(params["numSteps"]):
            for minimizer in minimizers:
                new_x = minimizer.step(positions[minimizer.label][-1])
                positions[minimizer.label].append(new_x)
                if minimizer.barycentric:
                    evaluations[minimizer.label].append(f_bar(new_x))
                else:
                    evaluations[minimizer.label].append(f(new_x))

    
    if params["plot"]:
        plot(triangle,minimizers,positions,f)