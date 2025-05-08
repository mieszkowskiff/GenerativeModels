import torch
import torch.nn as nn



class EncoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = True):
        super(EncoderConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.downsample:
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        return x



class Encoder(nn.Module):
    def __init__(self, dimension_list, latent_dim):
        super(Encoder, self).__init__()
        self.dimension_list = [(3, 64)] + dimension_list
        self.latent_dim = latent_dim
        self.blocks = nn.Sequential(
            *[
                EncoderConvBlock(
                    dimension_list[i][0], 
                    dimension_list[i+1][0], 
                    downsample = dimension_list[i + 1][1] < dimension_list[i][1]
                ) for i in range(len(dimension_list) - 1)
            ]
        )
        self.fc_mu = nn.Linear(self.dimension_list[-1][0] * self.dimension_list[-1][1], latent_dim)
        self.fc_logvar = nn.Linear(self.dimension_list[-1][0] * self.dimension_list[-1][1], latent_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = nn.Flatten()(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


