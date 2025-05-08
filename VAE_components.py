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
    def __init__(self, dimension_list, latent_dim, pic_size = 64):
        super(Encoder, self).__init__()
        self.dimension_list = [(3, pic_size)] + dimension_list
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
        self.fc_mu = nn.Linear(self.dimension_list[-1][0] * self.dimension_list[-1][1] * self.dimension_list[-1][1], latent_dim)
        self.fc_logvar = nn.Linear(self.dimension_list[-1][0] * self.dimension_list[-1][1] * self.dimension_list[-1][1], latent_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = nn.Flatten()(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    



class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample = True):
        super(DecoderConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = upsample
    
    def forward(self, x):
        if self.upsample:
            x = nn.Upsample(scale_factor=2)(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dimension_list, latent_dim, latent_sample_number = 1, pic_size = 64):
        super(Decoder, self).__init__()
        self.dimension_list = dimension_list + [(3, pic_size)]
        self.latent_dim = latent_dim
        self.latent_sample_number = latent_sample_number
        self.fc = nn.Linear(latent_dim * latent_sample_number, dimension_list[0][0] * dimension_list[0][1] * dimension_list[0][1])
        self.blocks = nn.Sequential(
            *[
                DecoderConvBlock(
                    dimension_list[i][0], 
                    dimension_list[i + 1][0], 
                    upsample = dimension_list[i + 1][1] > dimension_list[i][1]
                ) for i in range(len(dimension_list) - 1)
            ]
        )

    def forward(self, z):
        z = nn.Flatten()(z)
        z = self.fc(z)
        z = z.view(-1, self.dimension_list[0][0], self.dimension_list[0][1], self.dimension_list[0][1])
        for block in self.blocks:
            z = block(z)
        return nn.Sigmoid()(z)
    

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    



