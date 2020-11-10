import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# Input data's value range
lowest = 1e-9
highest = 1.0


class CustomModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = None
        self.R = None

    def relprop(self, R):
        self.R = R
        return R


# Define new layer including relevance propagation
class CustomReLU(CustomModule, nn.ReLU):
    def forward(self, input):
        self.x = input
        return super().forward(input)


class CustomSoftplus(CustomModule, nn.Softplus):
    def forward(self, input):
        self.x = input
        return super().forward(input)


class CustomPReLU(CustomModule, nn.PReLU):
    def forward(self, input):
        self.x = input
        return super().forward(input)


class CustomLeakyReLU(CustomModule, nn.LeakyReLU):
    def forward(self, input):
        self.x = input
        return super().forward(input)


class CustomLinear(CustomModule, nn.Linear):
    def forward(self, input):
        # restrict bias to negative values
        self.bias.data = torch.clamp(self.bias, max=0) - torch.clamp(self.bias, min=0) * 1e-4
        self.x = input
        return F.linear(input, self.weight, self.bias)


class NextLinear(CustomLinear):
    def relprop(self, R):
        weights = self.weight.detach().numpy()
        activation = self.x.detach().numpy()
        V = np.maximum(0, weights)
        Z = np.dot(activation, V.T)+1e-9
        S = R/Z
        C = np.dot(S, V)
        R = activation*C
        self.R = R
        return R


class FirstLinear(CustomLinear):
    def relprop(self, R):
        weights = self.weight.detach().numpy()
        activation = self.x.detach().numpy()
        W, V, U = weights, np.maximum(0, weights), np.minimum(0, weights)
        X, L, H = activation, activation * 0 + lowest, activation * 0 + highest
        Z = np.dot(X, W.T) - np.dot(L, V.T) - np.dot(H, U.T) + 1e-9
        S = R / Z
        R = X * np.dot(S, W) - L * np.dot(S, V) - H * np.dot(S, U)
        self.R = R
        return R


class CustomConv3d(CustomModule, nn.Conv3d):
    def forward(self, input):
        self.x = input
        self.bias.data = torch.clamp(self.bias, max=0) - torch.clamp(self.bias, min=0) * 1e-4
        return super().forward(input)


class NextConv3d(CustomConv3d):
    def relprop(self, R):
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float)

        pself = CustomConv3d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride)
        pself.bias.data *= 0
        pself.weight.data = torch.clamp_min(copy.deepcopy(self.weight), 0)

        x = copy.deepcopy(self.x.data).requires_grad_()
        Z = pself.forward(x)
        shape = Z.shape

        R = R.view(shape)
        S = R.squeeze().clone().detach() / (Z.squeeze() + 1e-9)

        Z.backward(S.view(shape))
        C = x.grad
        R = self.x*C
        self.R = R
        return R


class FirstConv3d(CustomConv3d):
    def relprop(self, R):
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float)

        iself = FirstConv3d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride)
        iself.bias.data *= 0
        iself.weight.data = copy.deepcopy(self.weight)

        nself = FirstConv3d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride)
        nself.bias.data *= 0
        nself.weight.data = np.minimum(0, copy.deepcopy(self.weight.data))

        pself = FirstConv3d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride)
        pself.bias.data *= 0
        pself.weight.data = np.maximum(0, copy.deepcopy(self.weight.data))

        X = copy.deepcopy(self.x.data).requires_grad_()
        L = (-torch.ones(self.x.shape) * lowest).requires_grad_()
        H = (torch.ones(self.x.shape) * highest).requires_grad_()

        Z1 = iself.forward(X)
        Z2 = pself.forward(L)
        Z3 = nself.forward(H)
        Z = Z1.data - Z2.data - Z3.data + 1e-9
        shape = Z.data.shape

        R = R.view(shape)
        S = R.squeeze().clone().detach() / Z.squeeze()

        Z1.backward(S.view(shape)+1e-9)
        C1 = X.grad
        Z2.backward(S.view(shape)+1e-9)
        C2 = L.grad
        Z3.backward(S.view(shape)+1e-9)
        C3 = H.grad

        R = X*C1 - L*C2 - H*C3
        self.R = R
        return R


class CustomMaxPool3d(CustomModule, nn.MaxPool3d):
    def forward(self, input):
        self.x = input
        return super().forward(input)

    def relprop(self, R):
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float)
        x = copy.deepcopy(self.x.data).requires_grad_()
        Z = self.forward(x)
        shape = Z.shape
        R = R.view(shape)
        S = R.squeeze().clone().detach() / (Z.squeeze() + 1e-9)
        # R.squeeze().clone().detach()
        Z.backward(S.view(shape))
        C = x.grad

        self.R = x * C
        return self.R


class CustomAvgPool3d(CustomModule, nn.AvgPool3d):
    def forward(self, input):
        self.x = input
        return super().forward(input)

    def relprop(self, R):
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float)
        x = copy.deepcopy(self.x.data).requires_grad_()
        Z = self.forward(x)
        shape = Z.shape
        R = R.view(shape)
        S = R.squeeze().clone().detach() / (Z.squeeze() + 1e-9)
        # R.squeeze().clone().detach()
        Z.backward(S.view(shape))
        C = x.grad

        self.R = x * C
        return self.R
