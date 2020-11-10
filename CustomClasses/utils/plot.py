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


# TODO: simplify filter_3d plotting
def make_ax(figsize, title='',  grid=False, fig=None):
    if fig is None:
        fig = plt.figure(title, figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    ax.axis('off')
    return ax


def return_colors(weights, m):
    dim = weights.shape[-1]
    assert dim == weights.shape[-2]
    colors = np.empty([dim, dim, dim, 4])
    for i in range(dim):
        colors[i] = m.to_rgba(weights[i])
    return colors


def filter3d(weight, filter_number, fig=None, figsize=(2, 2), title=''):
    weight = np.squeeze(weight)
    vmax = np.abs(weight).max()
    norm = plt.matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.bwr
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = return_colors(weight[filter_number], m)
    ax = make_ax(title=title, grid=True, figsize=figsize, fig=fig)
    full = np.ones(weight[filter_number].shape)
    ax.voxels(full, facecolors=colors, edgecolors='gray', alpha=0.9)
    plt.show()


def all_filter3d(model):
    for n_conv_layer in [model.index_conv_layer[0]]:
        conv_layer = model.layers[n_conv_layer]
        weight = conv_layer.weight.detach().numpy()
        for i_filter in range(conv_layer.out_channels):
            title = f'layer={n_conv_layer}, i_filter = {i_filter}'
            filter3d(weight, i_filter, title=title)