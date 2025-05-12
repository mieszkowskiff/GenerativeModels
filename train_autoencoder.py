import torch
import torch.nn as nn
from torchvision import transforms, datasets
from components import criterion, AutoEncoder, Encoder, Decoder
import tqdm
from torchsummary import summary
from utils import display

def main():
    torch.manual_seed(56)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root='data', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(
        latent_dim = 128
    )
    decoder = Decoder(
        latent_dim = 128
    )


    model = AutoEncoder(
        encoder = encoder, 
        decoder = decoder
    )

    model.to(device)

    summary(model, (3, 64, 64))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        for data, _ in tqdm.tqdm(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, z = model(data)
            loss = criterion(recon_batch, data, z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        with torch.no_grad():
            model.eval()
            loss = 0
            for data, _ in tqdm.tqdm(dataloader):
                data = data.to(device)
                recon_batch, z = model(data)
                loss += criterion(recon_batch, data, z)
            loss /= len(dataloader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    
    # Save the model
    torch.save(model.state_dict(), "autoencoder2.pth")
    with torch.no_grad():
        model.eval()
        for i in range(1000):
            img = dataloader.dataset[torch.randint(0, 1200, (1,))][0].unsqueeze(0).to(device)
            reconstructed, z = model(img)
            display(img)
            display(reconstructed)

        


            

        





if __name__ == "__main__":
    main()