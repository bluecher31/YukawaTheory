import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List

import CustomClasses.utils.utils_functions as utils
import CustomClasses.utils.plot as c_plt


def set_up_model(model: torch.nn.Module, ds: torch.utils.data.Dataset, kappa: float, g: float, batch_size: int):
    """
    performs forward and backward pass
    thereby all layers and nodes are assign a relevance internally
    """
    # forward pass
    field = ds.get_parameter_batch(kappa, g, batch_size=batch_size)
    _ = np.squeeze(model(torch.tensor(field))).detach().numpy()
    # backward pass
    relevance = ds.parameter_to_bins(kappa, g)
    _ = model.relprop(np.tile(relevance[np.newaxis], [batch_size, 1, 1]).reshape(batch_size, -1))


def get_index_conv_layers(model: torch.nn.Module) -> List[int]:
    """returns a index list of all convolutional layers"""
    list_conv_layer = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, torch.nn.Conv3d):
            list_conv_layer.append(i)
    return list_conv_layer


def error_handling_batch(filter_importance_batch: np.ndarray) -> np.ndarray:
    """Removes all samples from batch which have negative relevance"""
    # TODO: track down where falty samples come from
    if np.any(filter_importance_batch < 0):
        # print('WARNING: samples with negative filter_importance found!.')
        mask = filter_importance_batch < 0
        mask_correct_samples = (mask.sum(axis=0) == 0)
        filter_importance_batch = filter_importance_batch[:, mask_correct_samples]
    return filter_importance_batch


def get_filter_importance(model: torch.nn.Module, layer_number: int, batch_size: int) -> np.ndarray:
    """
    :param model: relevances are already precomputed
    :param layer_number: corresponds to convolution layer
    :param batch_size: which was used to precompute relevance
    :return: filter_relevance = np.array([[mean, error], [filer0,... , filter_n]]), shape = [2, n_filter]
    """
    conv = model.layers[layer_number]
    assert isinstance(conv, torch.nn.Conv3d)

    relevance_nodes = model.layers[layer_number + 1].R.reshape(batch_size, conv.out_channels, -1)
    if not isinstance(relevance_nodes, np.ndarray):
        relevance_nodes = relevance_nodes.detach().numpy()

    # calculate filter importance for all samples in batch
    filter_importance_batch = np.sum(relevance_nodes, axis=-1)      # average all relevances corresponding to a filter
    filter_importance_batch = np.swapaxes(filter_importance_batch, 1, 0)    # shape=(n_filters, batch_size)

    filter_importance_batch = error_handling_batch(filter_importance_batch)

    # use jackknife to to statistics over batch
    filter_importance = np.zeros((2, conv.out_channels))
    for i in range(conv.out_channels):
        filter_importance[0, i], filter_importance[1, i] = utils.jackknife(filter_importance_batch[i])
    filter_importance /= filter_importance[0].sum()     # normalize importance to unity
    return filter_importance


def get_local_model_importance(model: torch.nn.Module, ds: torch.utils.data.Dataset, kappa: float, g: float,
                               batch_size: int) -> List[np.ndarray]:
    """"returns importance for all filters in model for given parameter set"""
    set_up_model(model, ds, kappa, g, batch_size)

    # calculate filter relevance
    list_filter_importance = []
    for layer_number in model.index_conv_layer:
        filter_importance = get_filter_importance(model, layer_number, batch_size)
        list_filter_importance.append(filter_importance)

    return list_filter_importance


def get_global_model_importance(model: torch.nn.Module, ds: torch.utils.data.Dataset, batch_size: int) -> List[np.ndarray]:
    """
returns a list of filter importance for all conv layers
    :param model: _CustomeModel
    :param ds: YukawaDataset
    :return: List[filter_importance = np.array([[mean, error], n_filter, g, kappa])]
    """
    # preallocate memory and define container for all importances
    global_model_importance = []        # list np.arrays containing relevance for each conv layer
    for layer in model.index_conv_layer:    # loops all conv layers
        conv = model.layers[layer]
        global_model_importance.append(np.zeros(shape=(2, conv.out_channels, len(ds.g), len(ds.kappa))))

    for i_g, g in enumerate(ds.g):
        for j_kappa, kappa in enumerate(ds.kappa):
            list_filter_importance = get_local_model_importance(model, ds, kappa, g, batch_size=batch_size)
            for n_filter, filter_importance in enumerate(list_filter_importance):
                global_model_importance[n_filter][:, :, i_g, j_kappa] = filter_importance
    return global_model_importance


def plot_importance(model: torch.nn.Module, ds: torch.utils.data.Dataset, batch_size: int):
    global_model_importance = get_global_model_importance(model, ds, batch_size=batch_size)

    for n_conv_layer in range(len(model.index_conv_layer)):
        fig = plt.figure(f'Filter importance, n_conv_layer = {n_conv_layer}')

        filter_importance = global_model_importance[n_conv_layer]
        vmax = np.max(filter_importance[0])
        for n_filter in range(filter_importance.shape[1]):
            n_rows, n_cols = 4, 3
            fig.add_subplot(n_rows, n_cols, n_filter+1)
            _ = c_plt.imshow(filter_importance[0, n_filter], ds.kappa, ds.g,
                             colorbar=False, vmax=vmax, vmin=0, cmap='Greens')
            plt.axis('off')
            plt.tight_layout()

