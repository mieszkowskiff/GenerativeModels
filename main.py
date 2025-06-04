from components import DiffusionAutoencoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch
import utils



def main():
    torch.manual_seed(14)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4837, 0.4360, 0.3871), (.1994, 0.1982, 0.1976)),
        ])
    
    epochs = 100
    
    train_dataset = datasets.ImageFolder("./cats", transform = transform)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 4)



    
    model = DiffusionAutoencoder(latent_dim = 128, time_encoding_dim = 128, steps = 1000)

    for epoch in range(epochs):
        model.train()
        loss = 0
        for x, _ in tqdm(train_loader):
            loss += model.train_step(x)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

    model.eval()
    with torch.no_grad():
        for i in range(10):
            recon = model.sample()
            #print(recon)
            utils.display(recon)


    



if __name__ == "__main__":
    main()



