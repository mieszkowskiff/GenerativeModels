import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from utils.architecture import Generator

noise_dim = 100
num_images = 16
model_path = "./models/16th/generator.pth"  # Replace with your path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

for name, param in generator.named_parameters():
    print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
    break

z = (torch.randn(num_images, noise_dim, 1, 1)).to(device)

with torch.no_grad():
    fake_images = generator(z)

#mean = torch.tensor([0.48374429, 0.43603274, 0.38709232], device=fake_images.device).view(1, 3, 1, 1)
#std = torch.tensor([0.19943906, 0.19818559, 0.19755708], device=fake_images.device).view(1, 3, 1, 1)
mean = torch.tensor([0.5, 0.5, 0.5], device=fake_images.device).view(1, 3, 1, 1)
std = torch.tensor([0.5, 0.5, 0.5], device=fake_images.device).view(1, 3, 1, 1)
fake_images = fake_images * std + mean
fake_images = fake_images.clamp(0, 1)

fake_images_uint8 = (fake_images * 255).byte()

grid = vutils.make_grid(fake_images_uint8.cpu(), nrow=4, padding=2)

grid_image = grid.permute(1, 2, 0).numpy()

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(grid_image)
plt.show()
