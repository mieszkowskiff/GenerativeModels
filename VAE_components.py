import torch
import torch.nn as nn



class EncoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = True):
        super(EncoderConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = 3,
            padding = 1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.ReLU()(x)
        if self.downsample:
            x = nn.MaxPool2d(kernel_size = 2)(x)
        return x



class Encoder(nn.Module):
    def __init__(self, dimension_list, latent_dim, pic_size = 64):
        super(Encoder, self).__init__()
        self.dimension_list = dimension_list
        self.latent_dim = latent_dim
        self.blocks = nn.Sequential(
            *[
                EncoderConvBlock(
                    in_channels = self.dimension_list[i][0], 
                    out_channels = self.dimension_list[i][1], 
                    downsample = self.dimension_list[i][2],
                ) for i in range(len(self.dimension_list))
            ]
        )
        self.dense = nn.LazyLinear(512)
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_logvar = nn.LazyLinear(latent_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = nn.Flatten()(x)
        x = self.dense(x)
        x = torch.nn.ReLU()(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample = True):
        super(DecoderConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.upsample = upsample
    
    def forward(self, x):
        if self.upsample:
            x = nn.Upsample(scale_factor = 2)(x)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.ReLU()(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dimension_list, latent_dim, latent_sample_number = 1, pic_size = 64):
        super(Decoder, self).__init__()
        self.dimension_list = dimension_list
        self.latent_dim = latent_dim
        self.latent_sample_number = latent_sample_number
        self.dense = nn.Linear(latent_dim * latent_sample_number, 512)
        self.fc = nn.Linear(512, self.dimension_list[0][0])
        self.blocks = nn.Sequential(
            *[
                DecoderConvBlock(
                    in_channels = self.dimension_list[i][0], 
                    out_channels = self.dimension_list[i][1], 
                    upsample = self.dimension_list[i][2],
                ) for i in range(len(self.dimension_list))
            ]
        )
        assert pic_size == 2**sum(elem[2] for elem in self.dimension_list), "Wrong number of upsampling layers"

    def forward(self, z):
        z = nn.Flatten()(z)
        z = self.dense(z)
        z = torch.nn.ReLU()(z)
        z = self.fc(z)
        z = z.view(-1, self.dimension_list[0][0], 1, 1)
        for block in self.blocks:
            z = block(z)
        return nn.Tanh()(z) #nn.Sigmoid()(z)
    

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = encoder.latent_dim
        self.latent_sample_number = decoder.latent_sample_number
        assert self.latent_dim == decoder.latent_dim, "Encoder and decoder latent dimensions must match."


    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        batch_size, latent_dim = mu.size()

        # Generate multiple samples: (batch_size, latent_sample_number, latent_dim)
        eps = torch.randn(batch_size, self.latent_sample_number, latent_dim, device=mu.device)
        mu = mu.unsqueeze(1)  # (batch_size, 1, latent_dim)
        std = std.unsqueeze(1)  # (batch_size, 1, latent_dim)

        z = mu + eps * std  # (batch_size, latent_sample_number, latent_dim)

        # Concatenate along latent dimension
        z = z.view(batch_size, -1)  # (batch_size, latent_dim * latent_sample_number)
        return z


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    



def vae_loss_function(reconstructed, original, mu, logvar):
    # Rekonstrukcja: MSE lub BCE
    recon_loss = torch.nn.functional.mse_loss(reconstructed, original, reduction='sum')
    
    # KL-divergence (dla N(mu, sigma) || N(0, 1))
    # https://arxiv.org/abs/1312.6114 (Kingma & Welling)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_div