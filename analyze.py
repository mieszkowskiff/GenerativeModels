import torch
import VAE_components
import tqdm
from utils import display



def main():
    torch.manual_seed(453)


    latent_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE_components.VAE(
        encoder=VAE_components.Encoder(latent_dim = latent_dim),
        decoder=VAE_components.Decoder(latent_dim = latent_dim)
    )

    model.load_state_dict(torch.load(f"./models/VAEs/VAE.pth"))

    model.to(device)


    encoder = model.encoder
    decoder = model.decoder

    z1 = torch.randn(latent_dim).to(device)
    z2 = torch.randn(latent_dim).to(device)
    num_samples = 8
    decoder.to(device)
    with torch.no_grad():
        for i in range(num_samples):

            t = i / (num_samples - 1)
            z = (1 - t) * z1 + t * z2
            z = z.view(-1, latent_dim)
            display(decoder(z))

            





        




if __name__ == "__main__":
    main()