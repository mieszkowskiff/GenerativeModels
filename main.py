from components import MNISTDiffusionAutoencoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch
import utils



def main():
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    
    epochs = 10
    
    train_dataset = datasets.MNIST(root='./data', train = True, download = True, transform = transform)

    #indices = [i for i, (_, label) in enumerate(train_dataset) if label == 9]

    #Utwórz podzbiór tylko z siódemkami
    #subset = Subset(train_dataset, train_dataset)


    
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    model = MNISTDiffusionAutoencoder(latent_dim = 16, time_encoding_dim = 16, steps = 1000)
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



