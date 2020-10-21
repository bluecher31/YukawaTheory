import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import utils.plot as c_plt


def train(model: nn.Module, loader: torch.utils.data.DataLoader,
          learning_rate=1e-3, loss_function=nn.MSELoss(), break_after_fraction=1):

    model.train()
    n_batches = loader.__len__()
    print(f'Number of batches: {n_batches}.')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    train_loss = []
    for i, sample in enumerate(loader):
        optimizer.zero_grad()
        output = model(sample['configuration'])
        loss = loss_function(torch.squeeze(output), torch.squeeze(sample['label']))
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if i % 25 == 0 and i != 0:
            print(f"i = {i}, loss = {np.mean(train_loss[-25:]):.5F}")
        if i > np.floor(n_batches * break_after_fraction):
            print('Finished training earlier')
            break
    print(f"for the whole epoch loss = {np.mean(train_loss):.5F}")
    return np.mean(train_loss)


def measure_performance(model: nn.Module, ds: torch.utils.data.Dataset) -> Tuple[np.arange, np.array]:
    k_mean, g_mean = [], []
    for kappa, g in ds.labels:
        batch = ds.get_parameter_batch(kappa, g, batch_size=5)
        output = model(torch.tensor(batch))
        if output.shape[-1] > 2:        # binned regression
            k_prediction, g_prediction = ds.bins_to_parameter(output.detach().numpy())
        else:
            o = output.detach().numpy()
            k_prediction, g_prediction = o[:, 0], o[:, 1]
        k_mean.append(k_prediction.mean())
        g_mean.append(g_prediction.mean())

    k_mean = np.array(k_mean).reshape(len(ds.g), len(ds.kappa))
    delta_kappa = np.round(ds.kappa[0] - ds.kappa[1], 8)
    error_kappa = np.round((k_mean - ds.kappa) / delta_kappa, 8)

    g_mean = np.array(g_mean).reshape(len(ds.g), len(ds.kappa))
    delta_g = np.round(ds.g[0] - ds.g[1], 8)
    error_g = np.round((g_mean.T - ds.g) / delta_g, 8)
    return error_kappa, error_g.T


def visualize_performance(model: nn.Module, ds: torch.utils.data.Dataset):
    error_kappa, error_g = measure_performance(model, ds)

    vmax = int(len(ds.kappa)/2)
    plt.figure(f'Kappa Performance - test = {ds.load_test}')
    _ = c_plt.imshow(error_kappa, ds.kappa, ds.g, cmap='seismic', vmax=vmax, vmin=-vmax)

    vmax = int(len(ds.g) / 2)
    plt.figure(f'Yukawa Performance - test = {ds.load_test}')
    _ = c_plt.imshow(error_g, ds.kappa, ds.g, cmap='seismic', vmax=vmax, vmin=-vmax)
