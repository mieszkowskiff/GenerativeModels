from components import DiffusionAutoencoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch
import utils



mean = 0.
std = 0.
nb_samples = 0.


transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder("./cats", transform = transform)
loader = DataLoader(train_dataset, batch_size = 64, shuffle=True, num_workers = 4)


for data, _ in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f"Mean: {mean}")
print(f"Std: {std}")