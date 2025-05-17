from components import MNISTDiffusionAutoencoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    
    train_dataset = datasets.MNIST(root='./data', train = True, download = True, transform = transform)
    
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    model = MNISTDiffusionAutoencoder(latent_dim=2, time_encoding_dim=4)
    for epoch in range(10):
        model.train()
        loss = 0
        for x, _ in tqdm(train_loader):
            loss += model.train_step(x)
        print(f"Epoch {epoch + 1}, Loss: {loss}")



if __name__ == "__main__":
    main()



