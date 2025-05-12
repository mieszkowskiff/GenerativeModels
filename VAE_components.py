import torch


class EncoderConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_num = 3):
        super(EncoderConvBlock, self).__init__()

        self.residual = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1)

        self.conv = torch.nn.Sequential(
            *[
                item for _ in range(conv_num - 1) for item in (
                    torch.nn.Conv2d(
                        in_channels, 
                        in_channels, 
                        kernel_size = 3, 
                        padding = 1
                    ),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(in_channels),
                )
            ]
        )

        self.conv.append(
            torch.nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size = 3, 
                padding = 1
            )
        )
        
    def forward(self, x):

        residual = self.residual(x)
        x = self.conv(x)
        x += residual
        x = torch.nn.ReLU()(x)
        return torch.nn.MaxPool2d(kernel_size = 2)(x)


class Encoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = EncoderConvBlock(3, 32, conv_num = 2)
        self.conv2 = EncoderConvBlock(32, 64, conv_num = 2)
        self.conv3 = EncoderConvBlock(64, 128, conv_num = 2)
        self.conv4 = EncoderConvBlock(128, 256, conv_num = 2)

        self.dense = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 4 * 4, 256 * 4),
            torch.nn.ReLU(),
        )

        self.fc_mu = torch.nn.Linear(256 * 4, latent_dim)
        self.fc_logvar = torch.nn.Linear(256 * 4, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.dense(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

class DecoderConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_num = 3):
        super(DecoderConvBlock, self).__init__()

        self.residual = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1)

        self.conv = torch.nn.Sequential(
            *[
                item for _ in range(conv_num - 1) for item in (
                    torch.nn.ConvTranspose2d(
                        in_channels = in_channels,
                        out_channels = in_channels,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1
                    ),
                    torch.nn.ReLU(),
                )
            ]
        )

        self.conv.append(
            torch.nn.ConvTranspose2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 4,
                stride = 2,
                padding = 1
            )
        )

    def forward(self, x):
        residual = self.residual(x)
        residual = torch.nn.Upsample(scale_factor = 2)(residual)
        x = self.conv(x)
        x += residual
        x = torch.nn.ReLU()(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256 * 4),
            #torch.nn.LayerNorm(256 * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(256 * 4, 256 * 4 * 4),
            #torch.nn.LayerNorm(256 * 4 * 4),
            torch.nn.ReLU()
        )
        n = 5
        self.deconv1 = DecoderConvBlock(256, 128, conv_num = n)
        self.deconv2 = DecoderConvBlock(128, 64, conv_num = n)
        self.deconv3 = DecoderConvBlock(64, 32, conv_num = n)
        self.deconv4 = DecoderConvBlock(32, 3, conv_num = n)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)

        x = self.deconv1(x)
        x = self.deconv2(x)
        
        x = self.deconv3(x)
        
        x = self.deconv4(x)

        #print(x.min().item(), x.max().item())
        return torch.nn.Tanh()(x)
    
class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        ret = mu + eps * std
        
        #print(ret.min().item(), ret.max().item())
        return ret

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar




def vae_loss_function(recon_x, x, mu, logvar, split = False, beta = 1, multiplier = 1):
    """
    VAE loss function
    """
    BCE = torch.nn.functional.mse_loss(recon_x.view(-1), x.view(-1), reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    if split:
        return BCE , beta * KLD
    
    return multiplier * (BCE + beta * KLD)


