import torch



class EncoderConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, divider = 1, conv_num = 2):
        super(EncoderConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.divider = divider
        self.conv_num = conv_num

        self.residual_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = self.divider, padding = 0)

        self.convolutions = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1) for _ in range(self.conv_num - 1)
            ] + [
                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = self.divider, padding = 1)
            ]
        )

        self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.residual_conv(x)

        for conv in self.convolutions:
            x = conv(x)
            x = torch.nn.ReLU()(x)
        x = self.batch_norm(x)

        return torch.nn.ReLU()(x + residual)
        
        
class Encoder(torch.nn.Module):
    def __init__(self, conv_list, latent_dim = 256):
        super(Encoder, self).__init__()

        self.convs = torch.nn.Sequential(
            *[
                EncoderConvBlock(
                    in_channels = conv_desc["in_channels"],
                    out_channels = conv_desc["out_channels"],
                    divider = conv_desc["divider"],
                    conv_num = conv_desc["conv_num"]
                ) for conv_desc in conv_list
            ]
        )

        self.ffn = torch.nn.Sequential(
            torch.nn.LazyLinear(8 * latent_dim),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(4 * latent_dim),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(2 * latent_dim),
            torch.nn.ReLU()
        )

        self.mean = torch.nn.Linear(2 * latent_dim, latent_dim)
        self.log_var = torch.nn.Linear(2 * latent_dim, latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = torch.nn.Flatten()(x)
        x = self.ffn(x)

        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean, log_var
    

class DecoderConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, multiplier = 1, conv_num = 2):
        super(DecoderConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.multiplier = multiplier
        self.conv_num = conv_num

        self.residual_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)

        self.convolutions = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            ] + [
                torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1) for _ in range(self.conv_num - 1)
            ]
        )

        self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = torch.nn.Upsample(scale_factor = self.multiplier, mode = "bilinear", align_corners = True)(x)

        residual = self.residual_conv(x)

        for conv in self.convolutions:
            x = conv(x)
            x = torch.nn.ReLU()(x)
        #x = self.batch_norm(x)

        return torch.nn.ReLU()(x + residual)
    
class Decoder(torch.nn.Module):
    def __init__(self, conv_list, latent_dim = 256, pic_size = 64):
        super(Decoder, self).__init__()

        self.total_multiply = 1
        for conv_desc in conv_list:
            self.total_multiply *= conv_desc["multiplier"]

        self.initial_picture_size = pic_size // self.total_multiply

        self.ffn = torch.nn.Sequential(
            torch.nn.LazyLinear(2 * latent_dim),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(4 * latent_dim),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(8 * latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(8 * latent_dim, conv_list[0]["in_channels"] * (self.initial_picture_size ** 2)),
            torch.nn.ReLU()
        )

        self.convs = torch.nn.Sequential(
            *[
                DecoderConvBlock(
                    in_channels = conv_desc["in_channels"],
                    out_channels = conv_desc["out_channels"],
                    multiplier = conv_desc["multiplier"],
                    conv_num = conv_desc["conv_num"]
                ) for conv_desc in conv_list
            ]
        )

    
    def forward(self, x):

        x = self.ffn(x)
        x = x.view(x.shape[0], -1, self.initial_picture_size, self.initial_picture_size)
        x = self.convs(x)
        return torch.nn.Tanh()(x)

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean# + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, log_var


def vae_loss_function(recon_x, x, mu, logvar, split = False, beta = 0.01):
    MSE = torch.nn.MSELoss(reduction='sum')(recon_x, x)
    KLD = 0#-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    if split:
        return MSE, KLD
    return MSE + beta * KLD







