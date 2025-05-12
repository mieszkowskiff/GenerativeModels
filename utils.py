import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, -1, 1, 1)
    std = torch.tensor(std).reshape(1, -1, 1, 1)
    return tensor * std + mean

def display(picture):
    if not isinstance(picture, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    picture = picture.detach().cpu()
    #picture = denormalize(picture, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> (3, H, W)

    # Obsługa batcha = 1: (1, 3, H, W)
    if picture.ndim == 4 and picture.shape[0] == 1:
        picture = picture.squeeze(0)  # -> (3, H, W)

    if picture.ndim == 3 and picture.shape[0] == 3:
        picture = picture.permute(1, 2, 0)  # -> (H, W, 3)
    else:
        raise ValueError(f"Expected shape (3, H, W) or (1, 3, H, W), got {picture.shape}")

    plt.imshow(picture)
    plt.axis('off')
    plt.show()
