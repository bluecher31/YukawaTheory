import numpy as np
import matplotlib.pyplot as plt


def imshow(data: np.ndarray, xticks: np.ndarray, yticks: np.ndarray, colorbar=True, **kwargs):
    """
calls plt.imshow() and adds a plt.colorbar() plus axis labels
    :param data: [y, x]
    :param xticks:
    :param yticks:
    :param kwargs: additional keyword arguments for plt.imshow(), e.g. vmax, cmap
    :return: cmap
    """
    cmap = plt.imshow(data, **kwargs)

    plt.xlabel('$\kappa$')
    xsteps = int(len(xticks)/10)
    if xsteps == 0:
        xsteps = 1
    plt.xticks(np.arange(len(xticks))[::xsteps], xticks[::xsteps])

    plt.ylabel('g')
    ysteps = int(len(yticks)/6)
    if ysteps == 0:
        ysteps = 1
    plt.yticks(np.arange(len(yticks))[::ysteps], yticks[::ysteps])

    if colorbar:
        plt.colorbar(shrink=0.7)
    plt.tight_layout()
    return cmap
