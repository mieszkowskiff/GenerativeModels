import torch



class Encoder(torch.nn.Module):
    def _init__(self, latent_dim):
        super(Encoder, self)._init_()
        self.latent_dim = latent_dim
        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 4, 1024),
            torch.nn.ReLU(),

        )

        self.mu = torch.nn.Linear(1024, latent_dim)
        self.logvar = torch.nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = self.convolutions(x)
        x = self.dense(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256 * 4 * 4),
            torch.nn.ReLU()
        )

        self.deconvolutions = torch.nn.Sequential(
            torch.nn.Unflatten(1, (256, 4, 4)),
            torch.nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, z):
        x = self.dense(z)
        x = self.deconvolutions(x)
        return x
    
class VAE(torch.nn.Module):
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
    



