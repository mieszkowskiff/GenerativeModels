import torch
import math
import torchsummary

class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_encoding_dim = 4):
        super(ConvolutionalBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
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
        x = self.gn(x)
        x = self.activation(x)

        x = x * time_scale + time_shift

        return x
    

class ConvolutionalBlockTranspose(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_encoding_dim = 4):
        super(ConvolutionalBlockTranspose, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
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
        x = self.gn(x)
        x = self.activation(x)

        x = x * time_scale + time_shift

        return x
    
    
class MNISTAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim, time_encoding_dim = 4):
        super(MNISTAutoencoder, self).__init__()
        
        self.encoder_conv = torch.nn.ModuleList([
            ConvolutionalBlock(1, 16, time_encoding_dim),
            ConvolutionalBlock(16, 32, time_encoding_dim),
        ])

        self.dense = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 7 * 7, latent_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim * 4, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, latent_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim * 4, 32 * 7 * 7),
            torch.nn.Unflatten(1, (32, 7, 7)),
        )

        self.decoder_conv = torch.nn.ModuleList([
            ConvolutionalBlockTranspose(32, 16, time_encoding_dim),
            ConvolutionalBlockTranspose(16, 1, time_encoding_dim)
        ])

        self.latent_dim = latent_dim
        self.time_encoding_dim = time_encoding_dim

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_encoding_dim, time_encoding_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(time_encoding_dim * 4, time_encoding_dim),
            torch.nn.SiLU(),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def time_encoding(self, t):
        dim = self.time_encoding_dim
        half_dim = dim // 2


        exponent = -math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim) * exponent)
        angles = t[:, None] * freqs[None, :]

        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return emb

    def forward(self, x, t):
        t = self.time_encoding(t)
        t = t.to(self.device)
        t = self.time_mlp(t)
        

        for layer in self.encoder_conv:
            x = layer(x, t)

        x = self.dense(x)

        for layer in self.decoder_conv:
            x = layer(x, t)

        return x
    
    def fake_forward(self, x):
        t = torch.zeros(x.size(0), self.time_encoding_dim).to(self.device)
        for layer in self.encoder_conv:
            x = layer(x, t)

        x = self.dense(x)

        for layer in self.decoder_conv:
            x = layer(x, t)

        return x



class MNISTDiffusionAutoencoder(MNISTAutoencoder):
    def __init__(self, latent_dim = 2, time_encoding_dim = 2, beta_min = 0.001, beta_max = 0.02, steps = 1000):
        super(MNISTDiffusionAutoencoder, self).__init__(latent_dim = latent_dim, time_encoding_dim = time_encoding_dim)

        self.beta = torch.linspace(beta_min, beta_max, steps)
        self.alpha = 1 - self.beta

        self.alpha_hat = [1]
        for i in range(1, len(self.alpha)):
            self.alpha_hat.append(self.alpha_hat[i - 1] * self.alpha[i])
        self.alpha_hat.remove(1)

        self.alpha_hat = torch.tensor(self.alpha_hat)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.loss_fn = torch.nn.MSELoss()

        self.beta = self.beta.to(self.device)
        self.alpha_hat = self.alpha_hat.to(self.device)
        self.alpha = self.alpha.to(self.device)

        forward = self.forward

        self.forward = self.fake_forward

        torchsummary.summary(self, ((1, 28, 28)))

        self.forward = forward


    def train_step(self, x, t = None):
        x = x.to(self.device)
        if t is None:
            t = torch.randint(0, len(self.alpha) - 1, (x.size(0),)).long()
        noise = torch.randn_like(x).to(self.device)
        t = t.to(self.device)
        x_t = torch.sqrt(self.alpha_hat[t].reshape(-1, 1, 1, 1)) * x + torch.sqrt(1 - self.alpha_hat[t].reshape(-1, 1, 1, 1)) * noise
        t = t.to("cpu")
        output = self(x_t, t.float())
        loss = self.loss_fn(output, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

        


if __name__ == "__main__":
    a = MNISTDiffusionAutoencoder(latent_dim = 16, time_encoding_dim = 4)
    x = torch.randn(2, 1, 28, 28)
    a.train_step(x)
    