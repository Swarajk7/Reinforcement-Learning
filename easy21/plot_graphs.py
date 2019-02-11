from mpl_toolkits.mplot3d import Axes3D
from sys import platform as sys_pf
import numpy as np
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt


def plot_value_func(bestval):
    fig = plt.figure()
    ha = fig.add_subplot(111, projection='3d')
    x = range(21)
    y = range(10)
    X, Y = np.meshgrid(y, x)
    ha.plot_wireframe(X+1, Y+1, bestval[1:, 1:])
    ha.set_xlabel("dealer starting card")
    ha.set_ylabel("player current sum")
    ha.set_zlabel("value of state")
    plt.show()
