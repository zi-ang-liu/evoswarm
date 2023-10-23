import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from problems import *


if __name__ == '__main__':

    functions = [Ackley(), Rastrigin(), Sphere()]
    fig = plt.figure(figsize=(12, 4))
    for i, func in enumerate(functions):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        x = np.linspace(func.lower_bound[0], func.upper_bound[0], 100)
        y = np.linspace(func.lower_bound[1], func.upper_bound[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func.evaluate(np.array([X[i, j], Y[i, j]]))
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(func.name)
    plt.show()