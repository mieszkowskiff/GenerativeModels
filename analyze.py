import torch
import VAE_components
from torchvision import transforms, datasets
import tqdm
from utils import display
def main():
    torch.manual_seed(13)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(root='data', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE_components.VAE(
        encoder=VAE_components.Encoder(latent_dim=256),
        decoder=VAE_components.Decoder(latent_dim=256)
    )

    model.load_state_dict(torch.load("autoencoder2.pth"))

    model.to(device)


    encoder = model.encoder
    decoder = model.decoder

    with torch.no_grad():
            model.eval()
            MSE_loss = 0
            BCE_loss = 0
            for data, _ in tqdm.tqdm(dataloader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                mse, bce = VAE_components.vae_loss_function(recon_batch, data, mu, logvar, split=True)
                MSE_loss += mse
                BCE_loss += bce
            MSE_loss /= len(dataloader.dataset)
            BCE_loss /= len(dataloader.dataset)
            print(f"MSE_loss: {MSE_loss:.4f}, BCE_loss: {BCE_loss:.4f}")

    with torch.no_grad():
        for i in range(5):
            img = dataloader.dataset[torch.randint(0, 1200, (1,))][0].unsqueeze(0).to(device)
            reconstructed, mu, logvar = model(img)
            display(img)
            display(reconstructed)

    mus = torch.tensor([])
    encoder.eval()
    with torch.no_grad():
        for data, _ in tqdm.tqdm(dataloader):
            data = data.to(device)
            mu, logvar = encoder(data)
            mus = torch.cat((mus, mu.to("cpu")), dim=0)
    mean = mus.mean(dim=0)  # shape: [256]

    # 2. Centrowanie danych
    mus_centered = mus - mean

    # 3. Macierz kowariancji próbkowej
    N = mus.shape[0]
    cov = (mus_centered.T @ mus_centered) / (N - 1)

    eps = 1e-4
    cov_regularized = cov + eps * torch.eye(cov.shape[0], device=cov.device)
    L = torch.linalg.cholesky(cov_regularized).to(device)
    mean = mean.to(device)



    decoder.to(device)
    with torch.no_grad():
        for i in range(100):

            e = torch.randn(256).to(device)
            z = L @ e + mean
            z = z.view(-1, 256)

            display(decoder(z))

            





        




if __name__ == "__main__":
    main()