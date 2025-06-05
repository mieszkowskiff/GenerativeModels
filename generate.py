import torch
import components
from torchvision import transforms, datasets, utils
import tqdm
from utils import display

def main():

    latent_dim = 256
    torch.manual_seed(10)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(root = 'cats', transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = components.AutoEncoder(
        encoder=components.Encoder(latent_dim = latent_dim),
        decoder=components.Decoder(latent_dim = latent_dim)
    )

    model.load_state_dict(torch.load("./models/AutoEncoders/autoencoder256.pth"))

    model.to(device)


    encoder = model.encoder
    decoder = model.decoder

    model.eval()

    zs = torch.tensor([])
    encoder.eval()
    with torch.no_grad():
        for data, _ in tqdm.tqdm(dataloader):
            data = data.to(device)
            z = encoder(data)
            zs = torch.cat((zs, z.to("cpu")), dim=0)
    mean = zs.mean(dim=0)

    mus_centered = zs - mean

    N = zs.shape[0]
    cov = (mus_centered.T @ mus_centered) / (N - 1)

    eps = 1e-4
    cov_regularized = cov + eps * torch.eye(cov.shape[0], device=cov.device)
    L = torch.linalg.cholesky(cov_regularized).to(device)
    mean = mean.to(device)



    decoder.to(device)
    with torch.no_grad():
        for i in tqdm.tqdm(range(1000)):

            e = torch.randn(latent_dim).to(device)
            z = mean + L @ e
            z = z.view(-1, latent_dim)
            img = decoder(z).cpu()
            img = img.reshape((3, 64, 64))
            utils.save_image(
                img,
                f"./generated/AutoEncoder/{i:04d}.png",
                normalize = True,
                value_range = (-1, 1)
            )

            

            





        




if __name__ == "__main__":
    main()