import torch
import math
import tqdm
from transformer import DiffusionTransformer
from UNet import UNet



class DiffusionAutoencoder(DiffusionTransformer):
    def __init__(self, latent_dim = 128, time_encoding_dim = 128, beta_min = 0.001, beta_max = 0.02, steps = 1000):
        #super(DiffusionAutoencoder, self).__init__(latent_dim = latent_dim, time_encoding_dim = time_encoding_dim)
        super(DiffusionAutoencoder, self).__init__(
            d_embedding = latent_dim, 
            time_encoding_dim = time_encoding_dim,
            patch_size = 8,
            time_range = steps,
            n_transformer_blocks = 4,
            n_heads = 4,
            d_attention_hidden = latent_dim,
            d_ffn_hidden = latent_dim,
        )

        self.steps = steps

        self.beta = torch.linspace(beta_min, beta_max, steps)
        self.alpha = 1 - self.beta

        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.loss_fn = torch.nn.MSELoss()

        self.beta = self.beta.to(self.device)
        self.alpha_hat = self.alpha_hat.to(self.device)
        self.alpha = self.alpha.to(self.device)

        self.reconstructing_noise = False


    def train_step(self, x, t = None):
        x = x.to(self.device)
        if t is None:
            t = torch.randint(0, len(self.alpha) - 1, (x.size(0),)).long()
        noise = torch.randn_like(x).to(self.device)
        t = t.to(self.device)
        x_t = torch.sqrt(self.alpha_hat[t].reshape(-1, 1, 1, 1)) * x + torch.sqrt(1 - self.alpha_hat[t].reshape(-1, 1, 1, 1)) * noise

        output = self(x_t, t)
        if self.reconstructing_noise:
            loss = self.loss_fn(output, noise)
        else:
            loss = self.loss_fn(output, x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sample(self, x = None, T = None):
        if x is None:
            x = torch.randn((1, 3, 64, 64)).to(self.device)
        if T is None:
            T = self.steps - 1
        print("Sampling...")
        for t in tqdm.tqdm(range(T - 1, 0, -1)):
            t = torch.tensor([t]).to(self.device)
            if self.reconstructing_noise:
                recon_noise = self(x, t)
                
            else:
                recon_x = self(x, t)
                recon_noise  = x - recon_x
            mu = 1 / torch.sqrt(self.alpha[t]) * (x - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_hat[t] + 1e-5) * recon_noise)
            sigma = torch.sqrt(self.beta[t])
            x = mu + sigma * torch.randn_like(x)
            #x = torch.clamp(x, -3, 3)
        if self.reconstructing_noise:
            recon_noise = self(x, torch.tensor([0]).to(self.device)).to(self.device)
        else:
            recon_x = self(x, torch.tensor([0]).to(self.device)).to(self.device)
            recon_noise  = x - recon_x
        x = 1 / torch.sqrt(self.alpha[0]) * (x - (1 - self.alpha[0]) / torch.sqrt(1 - self.alpha_hat[0] + 1e-5) * recon_noise)
        #x = torch.clamp(x, -3, 3)
        return x

        

if __name__ == "__main__":
    # Test the MNISTDiffusionAutoencoder class
    model = DiffusionAutoencoder(latent_dim=16, time_encoding_dim = 2, steps = 1000)
    x = torch.randn((3, 1, 28, 28)).to(model.device)
    model.train()
    model.train_step(x)