import torch
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from utils.architecture import Generator

noise_dim = 100
model_path = "./models/pivot_8/generator.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(noise_dim=noise_dim).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

mean_tensor = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
std_tensor  = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

z = torch.randn(1, noise_dim, 1, 1, device=device)

fig = plt.figure(figsize=(6, 6))
ax_img   = fig.add_axes([0.1, 0.3, 0.8, 0.65])
ax_dim   = fig.add_axes([0.1, 0.2, 0.8, 0.03])
ax_value = fig.add_axes([0.1, 0.1, 0.8, 0.03])

slider_dim = Slider(ax_dim,   'Dim',   0, noise_dim-1, valinit=0, valstep=1)
slider_val = Slider(ax_value, 'Value', -4.0, 4.0, valinit=z[0,0,0,0].item())

def gen_image(z_tensor):
    with torch.no_grad():
        fake = generator(z_tensor).cpu()
    fake = (fake * std_tensor.cpu() + mean_tensor.cpu()).clamp(0,1)
    grid = vutils.make_grid(fake, nrow=1)
    return grid.permute(1,2,0).numpy()

img_disp = ax_img.imshow(gen_image(z), interpolation='nearest')
ax_img.axis('off')
ax_img.set_title('GAN Output')

def update(_):
    dim = int(slider_dim.val)
    val = slider_val.val
    #z_update = z.clone()
    #z_update[0, dim, 0, 0] = val
    #img_disp.set_data(gen_image(z_update))
    #z = z.clone()
    z[0, dim, 0, 0] = val
    img_disp.set_data(gen_image(z))
    
    fig.canvas.draw_idle()

slider_dim.on_changed(update)
slider_val.on_changed(update)

plt.show()
