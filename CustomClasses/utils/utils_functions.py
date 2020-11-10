import numpy as np
import torch
from typing import Tuple
import os


def gaussian(arr: np.array, mean=0.0, sigma=1.0):
    """
    :param arr: Array like, base value for which a gaussian is calculated
    :param mean: peak of gaussian
    :param sigma: variance of gaussian
    :return: gaussian(x): shape = arr.shape, gaussian is not normalized to unity
    """
    # result = 1/np.sqrt(1*np.pi*sigma**2)*np.exp(-(arr - mean)**2/(2*sigma**2))
    result = np.exp(-(arr - mean)**2/(2*sigma**2))
    return result


def jackknife(arr: np.array):
    if isinstance(arr, np.ndarray) is False:
        arr = np.array(arr.data)
    subs = []
    for i in range(len(arr)):
        subsample = np.delete(arr, i)
        subs.append(np.mean(subsample))
    subs = np.asarray(subs)
    mean = np.mean(subs)
    err = np.sqrt((len(arr)-1) * np.mean(np.square(subs - mean)))
    return mean, err


def store_model(model, path_to_folder: str):
    # check path_to_folder format
    if path_to_folder[-1] != '/':
        path_to_folder += '/'
    path_model = path_to_folder + 'parameters.pth'
    if os.path.exists(path_model):
        ValueError('do you really want to overwrite a saved model????')
        assert False
    torch.save(model, path_model)
    path = path_to_folder + 'NetworkArchitecture.txt'
    source_file = open(path, 'w')
    print(model, file=source_file)
    source_file.close()


def checkerboard(shape: np.ndarray) -> np.ndarray:
    mask = np.indices(shape).sum(axis=0) % 2
    mask[mask == 0] = -1
    return mask


mask_checkerboard = checkerboard(np.array([16, 16, 16]))
def get_staggered_magnetization(batch: np.ndarray) -> np.ndarray:
    assert len(batch.shape) > 3, 'please add an extra batch_size dimension'
    n_samples = batch.shape[0]
    observable = mask_checkerboard * batch
    observable = np.mean(observable.reshape(n_samples, -1), axis=1)
    return observable


def calculate_observables(ds: torch.utils.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    magnetization = np.zeros((len(ds.g), len(ds.kappa)))
    staggered_magnetization = np.zeros((len(ds.g), len(ds.kappa)))
    for i, g in enumerate(ds.g):
        for j, kappa in enumerate(ds.kappa):
            batch = ds.get_parameter_batch(kappa, g, batch_size=5)
            magnetization[i, j] = np.mean(batch)
            staggered_magnetization[i, j] = np.mean(get_staggered_magnetization(batch))
    return magnetization, staggered_magnetization
