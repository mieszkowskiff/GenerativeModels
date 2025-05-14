import torch
import torch.nn as nn
from torchvision import transforms, datasets
from VAE_components import VAE, vae_loss_function, Encoder, Decoder
import tqdm
from torchsummary import summary
from utils import display

def main():
    torch.manual_seed(43)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root='data', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(latent_dim = 128)
    decoder = Decoder(latent_dim = 128)



    model = VAE(
        encoder = encoder, 
        decoder = decoder
    )

    model.to(device)
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    _ = model(dummy_input)

    summary(model, (3, 64, 64))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        for data, _ in tqdm.tqdm(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            model.eval()
            MSE_loss = 0
            KLD_loss = 0
            for data, _ in tqdm.tqdm(dataloader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                mse, kld = vae_loss_function(recon_batch, data, mu, logvar, split = True)
                MSE_loss += mse
                KLD_loss += kld
            MSE_loss /= len(dataloader.dataset)
            KLD_loss /= len(dataloader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}, MSE_loss: {MSE_loss:.4f}, KLD_loss: {KLD_loss:.4f}")
            img = dataloader.dataset[torch.randint(0, 1200, (1,))][0].unsqueeze(0).to(device)
            reconstructed, mu, logvar = model(img)
            print(logvar.min().item(), logvar.max().item())

    
    # Save the model
    torch.save(model.state_dict(), "VAE.pth")
    with torch.no_grad():
        model.eval()
        for i in range(10):
            img = dataloader.dataset[torch.randint(0, 1200, (1,))][0].unsqueeze(0).to(device)
            reconstructed, mu, logvar = model(img)
            display(img)
            display(reconstructed)

    

        


            

        





if __name__ == "__main__":
    main()