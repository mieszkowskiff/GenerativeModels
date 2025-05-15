import torch
import torch.nn as nn
from torchvision import transforms, datasets
from VAE_components import VAE, LOSS, Encoder, Decoder
import tqdm
from torchsummary import summary
from utils import display

def main():
    torch.manual_seed(43)

    latent_dim = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root='cats', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(latent_dim = latent_dim)
    decoder = Decoder(latent_dim = latent_dim)



    model = VAE(
        encoder = encoder, 
        decoder = decoder
    )

    model.to(device)
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    _ = model(dummy_input)

    summary(model, (3, 64, 64))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 150
    model.train()
    for epoch in range(num_epochs):
        for data, _ in tqdm.tqdm(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = LOSS(recon_batch, data, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                model.eval()
                loss = 0
                for data, _ in tqdm.tqdm(dataloader):
                    data = data.to(device)
                    recon_batch, mu, logvar = model(data)
                    loss += LOSS(recon_batch, data, mu, logvar)
                loss /= len(dataloader.dataset)
                print(f"Epoch {epoch + 1}/{num_epochs}, LOSS: {loss.item()}")
                img = dataloader.dataset[torch.randint(0, 1200, (1,))][0].unsqueeze(0).to(device)
                reconstructed, mu, logvar = model(img)
                print(logvar.min().item(), logvar.max().item())

    
    # Save the model
    torch.save(model.state_dict(), "./models/VAEs/VAE.pth")
    with torch.no_grad():
        model.eval()
        for i in range(10):
            img = dataloader.dataset[torch.randint(0, 1200, (1,))][0].unsqueeze(0).to(device)
            reconstructed, mu, logvar = model(img)
            display(img)
            display(reconstructed)

    

        


            

        





if __name__ == "__main__":
    main()