import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <=3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')


def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m,'bias',None) is not None and bias_std is not None:
        if bias_std != 0: normal_(m.bias, 0, bias_std)
        else: m.bias.data.zero_()
    if init=='auto':
        if act_func in (F.relu_,F.leaky_relu_): init = kaiming_uniform_
        else: init = getattr(act_func.__class__, '__default_init__', None)
        if init is None: init = getattr(act_func, '__default_init__', None)
    if init is not None: init(m.weight)


class SpectralConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, init='auto', bias_std=0.01, **kwargs):
        if padding is None: padding = (ks-1)//2
        
        conv_func = _conv_func(ndim)
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        
        init_linear(conv, None, init=init, bias_std=bias_std)
        conv = spectral_norm(conv)
        layers = [conv]
       
        super().__init__(*layers)


class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super().__init__()
        self.query, self.key, self.value = [self._conv(n_channels, c) for c in (n_channels//8, n_channels//8, n_channels)]
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def _conv(self,n_in,n_out):
        return SpectralConvLayer(n_in, n_out, ks=1, ndim=1, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


# implementation below taken from
# https://github.com/sdoria/SimpleSelfAttention/blob/master/v0.1/Imagenette%20Simple%20Self%20Attention.ipynb

#Unmodified from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)



# Adapted from SelfAttention layer at https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
# Inspired by https://arxiv.org/pdf/1805.08318.pdf

class SimpleSelfAttention(nn.Module):
    def __init__(self, n_in: int, ks=1):  # , n_out:int):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks // 2, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        # self.sym = sym
        self.n_in = n_in

    def forward(self, x: torch.Tensor):
        size = x.size()
        x = x.view(*size[:2], -1)  # (C,N)

        convx = self.conv(x)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())  # (C,N) * (N,C) = (C,C)   => O(NC^2)

        o = torch.bmm(xxT, convx)  # (C,C) * (C,N) = (C,N)   => O(NC^2)

        o = self.gamma * o + x

        return o.view(*size).contiguous()