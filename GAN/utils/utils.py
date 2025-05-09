import torch
import torchvision.transforms as transforms

def save_txt(history, file_name):
    with open(file_name, "w") as f:
        for it in history:
            f.write(str(it) + " ")

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"