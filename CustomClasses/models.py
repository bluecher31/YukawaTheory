import torch
import torch.nn as nn
import numpy as np
from typing import List

import utils.lrp_architecture as lrp
from utils.self_attention import SelfAttention, SimpleSelfAttention

# base class used for all lrp models.
# additionally a forward pass and corresponding self.layers list needs to be initialised.


class _CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []

    def relprop(self, relevance: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[::-1]:
            relevance = layer.relprop(relevance)
        return relevance

    def _get_index_conv_layers(self) -> List[int]:
        """returns a index list of all convolutional layers"""
        list_conv_layer = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Conv3d):
                list_conv_layer.append(i)
        return list_conv_layer


class _Conv3dBlock(_CustomModel):
    def __init__(self, n_filters):
        super().__init__()
        self.conv = nn.Sequential(
            lrp.FirstConv3d(1, n_filters, 2),
            lrp.CustomReLU(),
            lrp.NextConv3d(n_filters, 5, 2),
            # SimpleSelfAttention(5),
            lrp.CustomReLU(),
            lrp.NextConv3d(5, 3, 2),
            lrp.CustomReLU(),
        )
        self.init_layers()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        N = input.shape[-1]         # lattice size
        return self.conv(input.view(batch_size, 1, N, N, N))

    def init_layers(self) -> None:
        for layer in self.conv.children():
            self.layers.append(layer)


class CNN_Classification(_CustomModel):
    def __init__(self, out_shape):
        super().__init__()
        n_filters = 8
        n_dense_nodes = 6591
        self.out_shape = out_shape
        assert isinstance(out_shape, np.ndarray), 'type(out_shape) is not np.ndarray'
        # n_output = n_kappa * n_g
        # self.n_kappa = n_kappa
        # self.n_g = n_g
        self.conv = _Conv3dBlock(n_filters)
        self.dense = nn.Sequential(
            lrp.NextLinear(n_dense_nodes, 2048),
            lrp.CustomReLU(),
            lrp.NextLinear(2048, 1024),
            lrp.CustomReLU(),
            lrp.NextLinear(1024, self.out_shape.prod()),
            lrp.CustomLeakyReLU(),
        )
        self.init_layers()
        self.index_conv_layer = self._get_index_conv_layers()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert isinstance(input, torch.Tensor), f'input hat incorrect type: {type(input)}'
        batch_size = input.shape[0]
        x = self.conv(input)
        x = x.view(batch_size, -1)
        x = self.dense(x)
        shape = tuple(np.insert(self.out_shape, 0, batch_size))
        return x.view(shape)

    def init_layers(self) -> None:
        self.layers = self.conv.layers
        for layer in self.dense.children():
            self.layers.append(layer)
