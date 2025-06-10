import torch
import VAE_components
import tqdm
from utils import display



def main():
    torch.manual_seed(7845)


    latent_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE_components.VAE(
        encoder=VAE_components.Encoder(latent_dim = latent_dim),
        decoder=VAE_components.Decoder(latent_dim = latent_dim)
    )

    model.load_state_dict(torch.load(f"./models/VAEs/VAE{latent_dim}.pth"))

    model.to(device)


    encoder = model.encoder
    decoder = model.decoder


    decoder.to(device)
    with torch.no_grad():
        for i in range(100):

            e = torch.randn(latent_dim).to(device)
            z = e
            z = z.view(-1, latent_dim)

            display(decoder(z))

            





        




if __name__ == "__main__":
    main()