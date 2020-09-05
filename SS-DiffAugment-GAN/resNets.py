import torch.nn as nn
from DiffAugment_pytorch import DiffAugment
from spectralNorm import SpectralNorm
import numpy as np
from torch.nn import utils

""" ----------  Generator ------------ """

class GenBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1, 
                activation=nn.ReLU(inplace=False), upsample=False):
        
        super(GenBlock, self).__init__()
        
        # Setting up important stuff:)
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        # Convolutional Layers
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight.data, 1.)
        nn.init.xavier_uniform_(self.c2.weight.data, 1.)

        # Batchnorm Layers
        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        nn.init.normal_(self.b1.weight.data, 1.0, 0.02)
        nn.init.constant_(self.b1.bias.data, 0.0)
        nn.init.normal_(self.b2.weight.data, 1.0, 0.02)
        nn.init.constant_(self.b2.bias.data, 0.0)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight.data, np.sqrt(2))
            if args.g_spectral_norm:
                self.c_sc = utils.spectral_norm(self.c_sc)

        if args.g_spectral_norm:
            print("Spectral on Generator.")
            self.c1 = utils.spectral_norm(self.c1)
            self.c2 = utils.spectral_norm(self.c2)

    def upsample_conv(self, x, conv):
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self.upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU(inplace=False)):
        super(Generator, self).__init__()
        self.bottom_width = args.bottom_width
        self.activation = activation
        self.ch = args.gf_dim
        
        # Specifying the necessary blocks.
        self.block2 = GenBlock(args, self.ch, self.ch // 2, activation=activation, upsample=True)
        self.block3 = GenBlock(args, self.ch // 2, self.ch // 4, activation=activation, upsample=True)
        self.block4 = GenBlock(args, self.ch // 4, self.ch // 8, activation=activation, upsample=True)
        #self.block5 = GenBlock(args, self.ch // 4, self.ch // 8, activation=activation, upsample=True)
        
        # Last BatchNorm + Convolutional and First Linear Layers
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.ch)
        self.b5 = nn.BatchNorm2d(self.ch // 8)
        self.c5 = nn.Conv2d(self.ch // 8, 3, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.l1.weight.data, 1.)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.)
        nn.init.normal_(self.b5.weight.data, 1.0, 0.02)
        nn.init.constant_(self.b5.bias.data, 0.0)

        # Adding Spectral Normalization to Generator
        if args.g_spectral_norm:
            self.l1 = utils.spectral_norm(self.l1)
            self.c5 = utils.spectral_norm(self.c5)

    def forward(self, z):
        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        #h = self.block5(h)
        h = self.b5(h)
        h = self.activation(h)
        h = nn.Tanh()(self.c5(h))
        return h


""" ----------  Discriminator ------------ """


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class OptimizedDisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU(inplace=False)):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        # Specifying the necessary blocks.
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        nn.init.xavier_uniform_(self.c1.weight.data, 1.)
        nn.init.xavier_uniform_(self.c2.weight.data, 1.)
        nn.init.xavier_uniform_(self.c_sc.weight.data, np.sqrt(2))
        
        # Adding spectral normalization
        if args.d_spectral_norm:
            print("Spectral on Discriminator.")
            self.c1 = utils.spectral_norm(self.c1)
            self.c2 = utils.spectral_norm(self.c2)
            self.c_sc = utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(inplace=False), downsample=False):
        
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        
        # Convolutional Layers
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight.data, 1.)
        nn.init.xavier_uniform_(self.c2.weight.data, 1.)
        
        # Adding Spectral Normalization
        if args.d_spectral_norm:
            self.c1 = utils.spectral_norm(self.c1)
            self.c2 = utils.spectral_norm(self.c2)

        # Corresponds to the 'stride =! 1' condition in original implementation.
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight.data, np.sqrt(2))
            if args.d_spectral_norm:
                self.c_sc = utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU(inplace=False), policy='color,translation,cutout'):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.num_classes = args.num_classes
        self.policy = policy
        
        # Specifying the necessary Blocks.
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch * 2, activation=activation, downsample=True)
        self.block3 = DisBlock(args, self.ch * 2, self.ch * 4, activation=activation, downsample=True)
        self.block4 = DisBlock(args, self.ch * 4, self.ch * 8, activation=activation, downsample=True)
        self.block5 = DisBlock(args, self.ch * 8, self.ch * 16, activation=activation, downsample=False)
        
        # Dense Layer at the end. Num_classes --> number of object classes (real) + 1 (fake)
        self.l5 = nn.Linear(self.ch * 16, self.num_classes, bias=False)
        nn.init.xavier_uniform_(self.l5.weight.data, 1.)
        
        # Adding Spectral Normalization to the Dense layer. 
        if args.d_spectral_norm:
            self.l5 = utils.spectral_norm(self.l5)

    def forward(self, x, feature=False, test=False):
        
        h = x
        # DiffAugment is applied here. 
        if test == False:
            h = DiffAugment(h, policy=self.policy)
        
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)
        if feature:
            return h, output
        return output



