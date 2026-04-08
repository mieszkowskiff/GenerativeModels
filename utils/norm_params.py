import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def compute_mean_std(dataset_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0

    print("Calculating mean and std...")

    for data, _ in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, 3, -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(f'\nDataset Mean: {mean.tolist()}')
    print(f'Dataset Std:  {std.tolist()}')

if __name__ == '__main__':
    dataset_path = '../dataset'  # change this
    compute_mean_std(dataset_path)
