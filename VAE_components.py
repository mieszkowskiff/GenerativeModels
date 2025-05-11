import torch


class EncoderConvBlock(torch.nn.Module):
    def __init__(self, channels):
        super(EncoderConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.bn = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return torch.nn.ReLU()(y + x)
    

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.starting_conv = torch.nn.Conv2d(
            in_channels = 3, 
            out_channels = 128, 
            kernel_size = 3, 
            padding = 1
        )

        self.conv = torch.nn.Sequential(
            EncoderConvBlock(128),
            EncoderConvBlock(128),
            EncoderConvBlock(128),
            EncoderConvBlock(128),
        )

        self.connection_conv = torch.nn.Conv2d(
            in_channels = 128, 
            out_channels = 4, 
            kernel_size = 3, 
            padding = 1,
            stride = 2
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 32 * 32, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
        )

        self.mean = torch.nn.Linear(1024, latent_dim)
        self.log_var = torch.nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = self.starting_conv(x)
        x = self.conv(x)
        x = self.connection_conv(x)
        x = torch.nn.ReLU()(x)
        x = self.dense(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class DecoderConvBlock(torch.nn.Module):
    def __init__(self, channels):
        super(DecoderConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.bn = torch.nn.BatchNorm2d(channels)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return torch.nn.ReLU()(x + y)
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 8192),
            torch.nn.ReLU(),
        )

        self.connection_conv = torch.nn.Conv2d(
            in_channels = 2, 
            out_channels = 128, 
            kernel_size = 3, 
            padding = 1
        )

        self.conv = torch.nn.Sequential(
            EncoderConvBlock(128),
            EncoderConvBlock(128),
            EncoderConvBlock(128),
            EncoderConvBlock(128),
        )

        self.ending_conv = torch.nn.Conv2d(
            in_channels = 128, 
            out_channels = 3, 
            kernel_size = 3, 
            padding = 1
        )

    def forward(self, z):
        z = self.dense(z)
        z = z.view(-1, 2, 64, 64)
        z = self.connection_conv(z)
        z = torch.nn.ReLU()(z)
        z = self.conv(z)
        z = self.ending_conv(z)
        return torch.nn.Tanh()(z)
        
    
class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, log_var


def vae_loss_function(recon_x, x, mu, logvar, split = False, alpha = 0.0001, beta = 0.01):
    MSE = torch.nn.MSELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    if split:
        return MSE, KLD
    return alpha * MSE + beta * KLD







