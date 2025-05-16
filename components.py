import autoencoder_components



class DiffusionAutoencoder(autoencoder_components.AutoEncoder):
    def __init__(self, latent_dim, beta_min = 0.001, beta_max = 0.02, steps = 1000):
        super(DiffusionAutoencoder, self).__init__(latent_dim)
        self.beta = list(range(beta_min, beta_max, (beta_max - beta_min) / steps))
        self.alpha = [1 - b for b in self.beta]
        self.alpha_hat = [1]
        for i in range(1, len(self.alpha)):
            self.alpha_hat.append(self.alpha_hat[i - 1] * self.alpha[i])


        