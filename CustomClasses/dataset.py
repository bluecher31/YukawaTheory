import os
import torch
# import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle   # potentially also use joblib or cPickle(_pickle)
from typing import List, Dict, Tuple

import utils.utils_functions as utils_functions
import utils.plot as c_plt

Tensor = torch.Tensor
Array = np.ndarray


class Error(Exception):
    pass


class YukawaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, kappa: Array, g: Array, M: float, load_test=False):
        self.M = M
        self.load_test = load_test
        if load_test:
            self.root_dir = f'{root_dir}/Test/M = {self.M:.0f}/'        # load small validation data
        else:
            self.root_dir = f'{root_dir}/Train/M = {self.M:.0f}/'       # load bigger training data
        self.kappa = np.round(np.sort(kappa), 8)            # ignore numerical noise
        self.kappa[self.kappa == 0] = 0.                     # set -0.000 to 0
        self.g = np.round(np.sort(g), 8)
        list_of_folders, list_of_labels = self._create_data_index()
        self.list_of_folders: List[str] = list_of_folders
        self.labels = np.array(list_of_labels)
        self.n_samples_per_folder = self._get_n_samples_per_folder()
        self._search_all_files()    # do all files exist?
        self.out_shape = np.array([2])
        print('Initialized YukawaDataset')

    def __len__(self) -> int:
        return int(len(self.list_of_folders) * self.n_samples_per_folder)

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        configuration, label = self._get_sample(item)
        sample = {'configuration': torch.tensor(configuration[np.newaxis].astype(np.float32)),
                  'label': torch.tensor(label.astype(np.float32))}
        return sample

    def get_parameter_batch(self, kappa: float, g: float, batch_size=10) -> np.ndarray:
        assert batch_size < self.n_samples_per_folder
        mask = np.logical_and(self.labels[:, 0] == kappa,
                              self.labels[:, 1] == g)
        if np.alltrue(mask == False):
            ValueError('parameter combination not found')
            assert mask.sum() == 1, f"kappa = {kappa:.4f}, g = {g:.4f} not contained in dataset"
        file_index = np.argmax(mask)
        index_list = np.arange(self.n_samples_per_folder)
        batch = []
        for i in np.random.permutation(index_list)[:batch_size]:
            item = file_index * self.n_samples_per_folder + i
            sample = self.__getitem__(item)
            batch.append(sample['configuration'].detach().numpy())
        return np.array(batch)

    def _get_sample(self, item: int) -> Tuple[Array, Array]:
        # map index to unique configuration
        folder_number, file_number = divmod(item, self.n_samples_per_folder)
        assert folder_number == int(folder_number)
        assert file_number == int(file_number)

        folder_path = self.list_of_folders[folder_number]
        configuration = pickle.load(open(folder_path + f"{file_number}", 'rb'))

        label = self.labels[folder_number]
        # label = self._convert_path_to_label(folder_path)
        return configuration, label

    def _get_n_samples_per_folder(self) -> int:
        n = np.random.randint(0, len(self.list_of_folders))
        folder_path = self.list_of_folders[n]
        if os.path.exists(folder_path):
            list_of_files = os.listdir(folder_path)
        else:
            raise Error(f"folder_path = {folder_path} does not exist.\n")
        n_samples_per_folder = len(list_of_files)
        return n_samples_per_folder

    def _convert_path_to_label(self, file_path: str) -> Array:
        splitted_path = file_path.split('/')
        kappa = float(splitted_path[-2][-6:])
        g = float(splitted_path[-3][-6:])
        return Array([kappa, g])

    def _create_data_index(self) -> Tuple[List[str], List[list]]:
        list_of_folders = []
        list_of_parameters = []
        # loop all parameters
        for g in self.g:
            for kappa in self.kappa:
                folder_path = self.root_dir + f"g = {g:.3f}/kappa = {kappa:.3f}/"
                list_of_folders.append(folder_path)  # store all paths
                list_of_parameters.append([kappa, g])   # store corresponding parameter pairs
        return list_of_folders, list_of_parameters

    def _search_all_files(self):    # only used for error catching
        for folder in self.list_of_folders:
            for i in range(self.n_samples_per_folder):
                file_path = folder + str(i)
                if not os.path.exists(file_path):
                    raise Error(f"file_path = {file_path} does not exist.\n"
                                f"[folder = {folder}] exist = {os.path.exists(folder)}")
        print('Found all data files.')


class YukawaDatasetClassification(YukawaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variance = 4
        self.out_shape = np.array([len(self.g), len(self.kappa)])
        print('Initialized YukawaDatasetClassification')

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        configuration, [kappa, g] = self._get_sample(item)
        label = self.parameter_to_bins(kappa, g)
        sample = {'configuration': torch.tensor(configuration[np.newaxis].astype(np.float32)),
                  'label': torch.tensor(label.astype(np.float32))}
        return sample

    def bins_to_parameter(self, bins: np.array) -> Tuple[float, float]:       # converts bins to Tuple[kappa, g]
        bins = bins[np.newaxis] if bins.ndim == 2 else bins
        batch_size = bins.shape[0]
        bins_flatten = bins.reshape(batch_size, -1)
        ind_g, ind_kappa = np.unravel_index(np.argmax(bins_flatten, axis=1), bins.shape[1:])
        return self.kappa[ind_kappa], self.g[ind_g]

    # converts Tuple[kappa, g] to bins with gaussian shapes
    def parameter_to_bins(self, kappa: float, g: float) -> np.array:
        mask_kappa = np.isclose(self.kappa, kappa)
        mask_g = np.isclose(self.g, g)
        assert mask_g.sum() == 1 and mask_kappa.sum() == 1, 'could not found parameter'      # check unique mapping
        kappa_max = mask_kappa.argmax()
        g_max = mask_g.argmax()

        bins_kappa = np.arange(len(self.kappa))
        bins_g = np.arange(len(self.g))
        kappa, g = np.meshgrid(bins_kappa, bins_g)
        bins = 10 * utils_functions.gaussian(kappa, mean=kappa_max, sigma=self.variance) \
               * utils_functions.gaussian(g, mean=g_max, sigma=self.variance)
        return bins


def plot_dataset(ds: YukawaDataset):
    magnetization, staggered_magnetization = utils_functions.calculate_observables(ds)
    vmax = np.abs(np.append(magnetization, staggered_magnetization)).max()
    plt.figure('Magnetization')
    c_plt.imshow(magnetization, ds.kappa, ds.g, cmap='seismic', vmax=vmax, vmin=-vmax)

    plt.figure('Staggered Magnetization')
    c_plt.imshow(staggered_magnetization, ds.kappa, ds.g, cmap='seismic', vmax=vmax, vmin=-vmax)
