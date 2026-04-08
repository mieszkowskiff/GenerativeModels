import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from utils.architecture import Generator
import torchvision

noise_dim = 100
num_images = 64
model = "NEW_5"
model_path = "./models/" + model + "/generator.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(noise_dim=noise_dim).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

for name, param in generator.named_parameters():
    print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")


z = (torch.randn(16, noise_dim, 1, 1)).to(device)

with torch.no_grad():
    print(z.shape)
    fakes = torch.stack([generator((torch.randn(1, noise_dim, 1, 1)*1).to(device)) for _ in range(num_images)], dim = 1)
    #fakes = generator(z)
    fakes = fakes[0]
    print(fakes.shape)
    mean_tensor = torch.tensor([0.5, 0.5, 0.5], device=fakes.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor([0.5, 0.5, 0.5], device=fakes.device).view(1, 3, 1, 1)
    print("GENERATED SHAPE:")
    print(fakes.shape)
    fakes = fakes.to(torch.float32)
    fakes = (fakes * std_tensor + mean_tensor).clamp(0, 1)
    grid = torchvision.utils.make_grid(fakes.cpu(), nrow=8)
    torchvision.utils.save_image(grid, f"results/gen_sample_{model}.png")

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(grid)
    plt.show()
