import torch
import torch.nn as nn
from torchvision import transforms, datasets
from VAE_components import VAE, vae_loss_function, Encoder, Decoder
import tqdm
from torchsummary import summary
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
    encoder = Encoder(
        conv_list = [
            {
                "in_channels": 3,
                "out_channels": 64,
                "divider": 2,
                "conv_num": 2
            },
            {
                "in_channels": 64,
                "out_channels": 128,
                "divider": 2,
                "conv_num": 2
            },
            {
                "in_channels": 128,
                "out_channels": 256,
                "divider": 2,
                "conv_num": 2
            }
        ]
    )
    decoder = Decoder(
        conv_list = [
            {
                "in_channels": 256,
                "out_channels": 128,
                "multiplier": 2,
                "conv_num": 2
            },
            {
                "in_channels": 128,
                "out_channels": 64,
                "multiplier": 2,
                "conv_num": 2
            },
            {
                "in_channels": 64,
                "out_channels": 3,
                "multiplier": 2,
                "conv_num": 2
            }
        ]
    )



    model = VAE(
        encoder = encoder, 
        decoder = decoder
    )

    model.to(device)

    summary(model, (3, 64, 64))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50
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
            BCE_loss = 0
            for data, _ in tqdm.tqdm(dataloader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                mse, bce = vae_loss_function(recon_batch, data, mu, logvar, split=True)
                MSE_loss += mse
                BCE_loss += bce
            MSE_loss /= len(dataloader.dataset)
            BCE_loss /= len(dataloader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}, MSE_loss: {MSE_loss:.4f}, BCE_loss: {BCE_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "vae_model.pth")
    with torch.no_grad():
        model.eval()
        for i in range(1000):
            img = dataloader.dataset[torch.randint(0, 1200, (1,))][0].unsqueeze(0).to(device)
            reconstructed, mu, logvar = model(img)
            display(img)
            display(reconstructed)

        


            

        





if __name__ == "__main__":
    main()