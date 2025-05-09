import torch
import torch.nn as nn
from torchvision import transforms, datasets
from VAE_components import VAE, vae_loss_function, Encoder, Decoder
import tqdm
from torchsummary import summary
from utils import display

def main():
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root='data', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(
        latent_dim = 128, 
        dimension_list = [
            (32, 64), 
            (64, 32), 
            (128, 16),
            (256, 8)
        ]
    )
    decoder = Decoder(
        latent_dim = 128, 
        dimension_list = [
            (256, 8),
            (128, 16), 
            (64, 32), 
            (32, 64)
        ], 
        latent_sample_number = 1
    )


    model = VAE(
        encoder = encoder, 
        decoder = decoder
    )

    model.to(device)

    summary(model, (3, 64, 64))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        for data, _ in tqdm.tqdm(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            loss = 0
            for data, _ in tqdm.tqdm(dataloader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss += vae_loss_function(recon_batch, data, mu, logvar)
            loss /= len(dataloader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
            
            img = dataloader.dataset[torch.randint(0, 1200, (1,))][0].unsqueeze(0).to(device)
            reconstructed, mu, logvar = model(img)
            print(mu)
            print(logvar)
            display(img)
            display(reconstructed)

        


            

        





if __name__ == "__main__":
    main()