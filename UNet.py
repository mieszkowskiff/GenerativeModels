import torch
import math


class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_encoding_dim = 128, downsample = True):
        super(ConvolutionalBlock, self).__init__()
        self.downsample = downsample
        if downsample:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        else:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.gn = torch.nn.GroupNorm(1, out_channels)
        self.time_scale = torch.nn.Linear(time_encoding_dim, out_channels)
        self.time_shift = torch.nn.Linear(time_encoding_dim, out_channels)

        self.activation = torch.nn.SiLU()

    def forward(self, x, time_encoding):
        time_scale = self.time_scale(time_encoding).unsqueeze(-1).unsqueeze(-1)
        time_shift = self.time_shift(time_encoding).unsqueeze(-1).unsqueeze(-1)

        time_scale = self.activation(time_scale)
        time_shift = self.activation(time_shift)


        
        x = self.conv(x)
        
        x = self.activation(x)
        x = self.gn(x)

        x = x * time_scale + time_shift

        return x
    

class ConvolutionalBlockTranspose(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_encoding_dim = 128, upsample = True, skip_connection_channels = 0):
        super(ConvolutionalBlockTranspose, self).__init__()
        self.upsample = upsample
        if upsample:
            self.conv = torch.nn.ConvTranspose2d(
                in_channels + skip_connection_channels, 
                out_channels, 
                kernel_size = 4,
                stride = 2, 
                padding = 1
                )
        else:
            self.conv = torch.nn.ConvTranspose2d(
                in_channels + skip_connection_channels, 
                out_channels, 
                kernel_size = 3, 
                padding = 1
                )
        self.gn = torch.nn.GroupNorm(1, out_channels)
        self.time_scale = torch.nn.Linear(time_encoding_dim, out_channels)
        self.time_shift = torch.nn.Linear(time_encoding_dim, out_channels)

        self.activation = torch.nn.SiLU()


    def forward(self, x, time_encoding, skip_connection):

        time_scale = self.time_scale(time_encoding).unsqueeze(-1).unsqueeze(-1)
        time_shift = self.time_shift(time_encoding).unsqueeze(-1).unsqueeze(-1)

        time_scale = self.activation(time_scale)
        time_shift = self.activation(time_shift)

        x = torch.cat([x, skip_connection], dim = 1)

        
        x = self.conv(x)
        x = self.activation(x)
        x = self.gn(x)

        x = x * time_scale + time_shift

        return x
    
    
class UNet(torch.nn.Module):
    def __init__(self, latent_dim, time_encoding_dim = 128):
        super(UNet, self).__init__()
        
        self.encoder_conv = torch.nn.ModuleList([
            ConvolutionalBlock(3, 32, time_encoding_dim),
            ConvolutionalBlock(32, 64, time_encoding_dim),
            ConvolutionalBlock(64, 128, time_encoding_dim),
            ConvolutionalBlock(128, 256, time_encoding_dim),
            ConvolutionalBlock(256, 512, time_encoding_dim),
            ConvolutionalBlock(512, 1024, time_encoding_dim),
        ])

        self.dense = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.SiLU(),
            torch.nn.Linear(1024 * 1 * 1, latent_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(latent_dim * 4, latent_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(latent_dim, latent_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(latent_dim * 4, 1024 * 1 * 1),
            torch.nn.SiLU(),
            torch.nn.Unflatten(1, (1024, 1, 1))
        )


        self.decoder_conv = torch.nn.ModuleList([
            ConvolutionalBlockTranspose(1024, 512, time_encoding_dim, skip_connection_channels = 1024),
            ConvolutionalBlockTranspose(512, 256, time_encoding_dim, skip_connection_channels = 512),
            ConvolutionalBlockTranspose(256, 128, time_encoding_dim, skip_connection_channels = 256),
            ConvolutionalBlockTranspose(128, 64, time_encoding_dim, skip_connection_channels = 128),
            ConvolutionalBlockTranspose(64, 32, time_encoding_dim, skip_connection_channels = 64),
            ConvolutionalBlockTranspose(32, 3, time_encoding_dim, skip_connection_channels = 32),
        ])

        self.latent_dim = latent_dim
        self.time_encoding_dim = time_encoding_dim

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_encoding_dim, time_encoding_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(time_encoding_dim * 4, time_encoding_dim),
            torch.nn.SiLU(),
        )

        self.final_activation = torch.nn.Tanh()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def time_encoding(self, t):
        sin = torch.cat([torch.sin(2 * math.pi * t * mul / self.steps).unsqueeze(1) for mul in range(1, self.time_encoding_dim // 2 + 1)], dim = 1)
        cos = torch.cat([torch.cos(2 * math.pi * t * mul / self.steps).unsqueeze(1) for mul in range(1, self.time_encoding_dim // 2 + 1)], dim = 1)
        
        out = torch.cat([sin, cos], dim = 1)
        return out
    
    def forward(self, x, t):
        time_encoding = self.time_encoding(t)
        time_encoding = self.time_mlp(time_encoding)

        skip_connections = []
        for idx, conv in enumerate(self.encoder_conv):
            x = conv(x, time_encoding)
            skip_connections.append(x.clone())

        x = self.dense(x)

        for idx, conv in enumerate(self.decoder_conv):
            x = conv(x, time_encoding, skip_connections[-(idx + 1)])

        return x