import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
import numpy as np
from scipy.spatial import distance

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(self.weight.size())
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class repux_block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, ks, a, drop_path=0., layer_scale_init_value=1e-6, deploy=False):
        super().__init__()
        
        ## Block Structure
        self.ks = ks
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=self.ks, padding=self.ks//2, groups=dim)
        self.norm = nn.BatchNorm3d(dim)
        self.act = nn.GELU()
        self.deploy = deploy

        ## Bayesian Frequency Matrix (BFM) 
        if self.deploy == False:
            alpha = a
            BFM = np.zeros((dim, 1, self.ks, self.ks, self.ks))
            for i in range(self.ks):
                for j in range(self.ks):
                    for k in range(self.ks):
                        point_1 = (i, j, k)
                        point_2 = (self.ks//2, self.ks//2, self.ks//2)
                        dist = distance.euclidean(point_1, point_2)
                        BFM[:, :, i, j, k] = alpha / (dist + alpha)
                        self.BFM = torch.from_numpy(BFM).type(torch.cuda.FloatTensor)

    def forward(self, x):
        ## Re-parameterize the convolutional layer weights
        if self.deploy == False:  ## Only perform re-parameterization in training
            w_0 = self.dwconv.weight
            w_1 = w_0 * self.BFM
            self.dwconv.weight = torch.nn.Parameter(w_1)

        ## Perform Convolution-Norm-Activation
        feat = self.dwconv(x)
        feat = self.norm(feat)
        feat = self.act(feat)

        return feat


class repuxnet_conv(nn.Module):
    """
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], ks=21, a=1,
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], deploy=False):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # stem = nn.Sequential(
        #     nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.deploy = deploy
        for i in range(4):
            stage = nn.Sequential(
                *[repux_block(dim=dims[i], ks=ks, a=a, drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, deploy=self.deploy) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # self.apply(self._init_weights)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            # print(i)
            # print(x.size())
            x = self.downsample_layers[i](x)
            # print(x.size())
            x = self.stages[i](x)
            # print(x.size())
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

