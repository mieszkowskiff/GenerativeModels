import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    subset_indices = torch.where(mnist_train.targets == 7)[0]
    mnist_subset = torch.utils.data.Subset(mnist_train, subset_indices)
    subset_loader = DataLoader(mnist_subset, batch_size=64, shuffle=True)

    for images, _ in subset_loader:
        print(images.shape)




if __name__ == "__main__":
    main()



