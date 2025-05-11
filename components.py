import torch
import torch.nn as nn


class EncoderConvBlock(nn.Module):
    def __init__(self, channels):
        super(EncoderConvBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return nn.ReLU()(y + x)
    

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.starting_conv = nn.Conv2d(
            in_channels = 3, 
            out_channels = 128, 
            kernel_size = 3, 
            padding = 1
        )

        self.conv = nn.Sequential(
            EncoderConvBlock(128),
            EncoderConvBlock(128),
            EncoderConvBlock(128),
            EncoderConvBlock(128),
        )

        self.connection_conv = nn.Conv2d(
            in_channels = 128, 
            out_channels = 2, 
            kernel_size = 3, 
            padding = 1
        )

        self.dense = torch.nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(latent_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.starting_conv(x)
        x = self.conv(x)
        x = self.connection_conv(x)
        x = nn.ReLU()(x)
        x = self.dense(x)
        return x


class DecoderConvBlock(nn.Module):
    def __init__(self, channels):
        super(DecoderConvBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return nn.ReLU()(x + y)
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.dense = torch.nn.Sequential(
            nn.LazyLinear(latent_dim),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.LazyLinear(8192),
            nn.ReLU(),
        )

        self.connection_conv = nn.Conv2d(
            in_channels = 2, 
            out_channels = 128, 
            kernel_size = 3, 
            padding = 1
        )

        self.conv = nn.Sequential(
            EncoderConvBlock(128),
            EncoderConvBlock(128),
            EncoderConvBlock(128),
            EncoderConvBlock(128),
        )

        self.ending_conv = nn.Conv2d(
            in_channels = 128, 
            out_channels = 3, 
            kernel_size = 3, 
            padding = 1
        )

    def forward(self, z):
        z = self.dense(z)
        z = z.view(-1, 2, 64, 64)
        z = self.connection_conv(z)
        z = nn.ReLU()(z)
        z = self.conv(z)
        z = self.ending_conv(z)
        return z
        
    

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = encoder.latent_dim
        assert self.latent_dim == decoder.latent_dim, "Encoder and decoder latent dimensions must match."



    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z

    
def criterion(recon_batch, data, z):
    return nn.MSELoss(reduction = 'sum')(recon_batch, data) + 0.1 * nn.MSELoss(reduction = 'sum')(z, torch.zeros_like(z))