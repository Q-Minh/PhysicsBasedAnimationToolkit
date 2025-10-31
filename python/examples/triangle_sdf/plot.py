import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable

from .minimizer import Minimizer

def plot(triangle: np.ndarray, minimizers: list[Minimizer], positions: dict, sdf_f: Callable[[np.ndarray], float]):
    bounding_box = np.array([[np.min(triangle[:,0]), np.min(triangle[:,1]), np.min(triangle[:,2])],
                                 [np.max(triangle[:,0]), np.max(triangle[:,1]), np.max(triangle[:,2])]])
    padded_box = bounding_box + np.array([[-1,-1,-1],[1,1,1]])
    A, B, C = triangle
    DX = np.vstack([B - A, C - A]).T

    n = np.cross(B - A, C - A)/np.linalg.norm(np.cross(B - A, C - A))
    d = -np.dot(A ,n)


    x_range = np.arange(padded_box[0,0],padded_box[1,0], 0.05)
    y_range = np.arange(padded_box[0,1],padded_box[1,1], 0.05)
    
    vec_f = np.vectorize(lambda x,y,z: sdf_f(np.array([x,y,z])))
    X, Y = np.meshgrid(x_range,y_range)
    Z = (-n[0] * X - n[1] * Y -d) /n[2]
    W =vec_f(X,Y,Z)
    minn, maxx = W.min(), W.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(W)



    ax = plt.figure().add_subplot(projection="3d",computed_zorder=False)

    for label, data in positions.items():
        minimizer = next((m for m in  minimizers if m.label == label), None)
        data = np.array(data)
        
        if minimizer.barycentric:
            pos = np.array(list(map(lambda p: DX @ p + A, data)))
        else:
            pos = data
        ax.plot(pos[:,0], pos[:,1], pos[:,2], marker=".", label=label, zorder=1.3)

    ax.plot_trisurf(triangle[:,0],triangle[:,1], triangle[:,2], color=(0,0,0,0), edgecolor="white", linewidth=1, antialiased=True, zorder=1.2)
    ax.plot_surface(X,Y,Z,facecolors=fcolors, vmin=minn, vmax=maxx, zorder=1.1 )

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()