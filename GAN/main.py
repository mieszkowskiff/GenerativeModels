import torch
import time
import tqdm
import copy
import shutil
import numpy as np
import torchvision

from torchsummary import summary
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import utils.components as components
import utils.utils as utils
from utils.architecture import Generator
from utils.architecture import Discriminator

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch
import os

import math
import matplotlib.pyplot as plt

# original cats dataset params
#mean=[0.4837442934513092, 0.4360327422618866, 0.38709232211112976]
#std=[0.19943906366825104, 0.19818559288978577, 0.19755707681179047]
# [-1, 1] normalization approach
mean=[0.5, 0.5, 0.5]
std=[0.5, 0.5, 0.5]

def plot_layer_heatmaps(grad_means: dict, title: str, save_path: str):
    layers = list(grad_means.keys())
    n = len(layers)

    width_ratios = []
    for name in layers:
        gm = grad_means[name]
      
        size = gm.numel()
        width_ratios.append(size)

    total = sum(width_ratios)
    width_ratios = [w/total for w in width_ratios]

    fig = plt.figure(figsize=(n*3, 4))
    gs = fig.add_gridspec(1, n, width_ratios=width_ratios, wspace=0.4)

    for i, name in enumerate(layers):
        ax = fig.add_subplot(gs[0, i])
        gm = grad_means[name].cpu().numpy()
        if gm.ndim > 2:
            gm = gm.reshape(gm.shape[0], -1)
        im = ax.imshow(gm, aspect='auto')
     
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        ax.set_title(name, fontsize=10)
        ax.axis('off')
    fig.suptitle(title, fontsize=14)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def diversity_loss_pixel(x):

    B = x.shape[0]

    flat = x.view(B, -1)
  
    var_per_dim = flat.var(dim=0)
    return var_per_dim.mean()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 150
    batch_size = 256
    noise_dim = 100

    p = 0.15
    eps = 0.01

    lam_gap = 0.3
    fixed_z = torch.randn(16, noise_dim, 1, 1).to(device)
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(device)

 
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5), 
            std=(0.5, 0.5, 0.5)
        )
    ])

    train_dataset = ImageFolder(root='../dataset', transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # real images loaded
    # initialize the Generator and Discriminator

    generator = Generator(noise_dim=noise_dim)
    generator.to(device)
    gen_parts = [generator.conv1, generator.bn1, generator.noise1, 
                 generator.conv2, generator.bn2, generator.noise2,
                 generator.conv3, generator.bn3, generator.noise3,
                 generator.conv4, generator.bn4, generator.noise4,
                 generator.refine_conv, generator.refine_bn, generator.noise_refine,
                 generator.conv4, generator.bn4, generator.noise4,
                 generator.conv6]
    discriminator = Discriminator()
    discriminator.to(device)

    # different loss functions ???
    criterion = torch.nn.BCEWithLogitsLoss()
    #discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
    
    #################################
    
    #low_lr_params = list(discriminator.init_block.parameters()) + list(discriminator.down0.parameters())
    low_lr_params = [discriminator.init_block, discriminator.down0]
    
    #all_params = set([discriminator.parameters()])
    
    #high_lr_params = [p for p in all_params if p not in low_lr_params]
    high_lr_params = [
                        discriminator.down1, 
                        discriminator.down2, 
                        discriminator.down3, 
                        discriminator.gap,
                        discriminator.classifier
                    ]
    
    low = []
    for it in low_lr_params:
        low += it.parameters()
    
    high = []
    for it in high_lr_params:
        high += it.parameters()
    
        
    discriminator_optimizer = torch.optim.Adam([
        {
        'params': low,
        # 1e-4
        'lr': 5e-4,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-4
        },
        {
        'params': high,
        'lr': 2e-4,
        'betas': (0.5, 0.999),
        'weight_decay': 1e-4
        }
    ])

    #                                                                   1e-4
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr = 0.0005, betas=(0.5, 0.99))

    generator_scaler = GradScaler()
    discriminator_scaler = GradScaler()

    for epoch in range(epochs):
        start_time = time.time()

        discriminator.train()
        generator.train()
        discriminator_total_loss = 0
        generator_total_loss = 0
        counter = 0
        
        for images, _ in tqdm.tqdm(train_loader):
            # real
            real_images = images.to(device)
            
            # it was 0.05, make discriminator more tolerant
            real_images += 0.05 * torch.randn_like(real_images)
            real_images = real_images.clamp(-1, 1)
            
            # smoothing out the label, using 0.9 instead of 1.0
            # this is done in order to prevent the discriminator from full confidence
            real_labels = torch.full((real_images.shape[0], 1), 0.9, device=device)

            # fake
            z = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            #generator.eval()
            fake_images = generator(z)
            # ??? maybe trying out the smooth labels here also ???
            fake_labels = torch.zeros((batch_size, 1), device=device)
            #fake_labels = torch.full((fake_images.shape[0], 1), 0.1, device=device)

            #########################
            # train the discriminator
            #discriminator.train()

            # Approach where optimizer step is done separately for real and fake images
            discriminator_optimizer.zero_grad()
            d_loss = 0

            if counter % 1 == 0:
                with autocast(device_type='cuda'):
                    outputs = discriminator(real_images)
                    # it was 1.2, focus classificator on understanding real cats, ad rejecting noise
                    # not vice versa, extract real abstract cat features
                    loss = criterion(outputs, real_labels) * 1.3
                
                d_loss += loss            
                discriminator_scaler.scale(loss).backward()
                discriminator_scaler.step(discriminator_optimizer)
                discriminator_scaler.update()
                discriminator_total_loss += loss.item()
                '''
                if counter % 10:
                    for name, p in discriminator.named_parameters():
                        if p.grad is not None:
                            disc_grad_sums[name] += p.grad.abs()
                    disc_batches += 1
                '''
                discriminator_optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    outputs = discriminator(fake_images)
                    loss = criterion(outputs, fake_labels)
                
                d_loss += loss
                discriminator_scaler.scale(loss).backward()
                discriminator_scaler.step(discriminator_optimizer)
                discriminator_scaler.update()
                discriminator_total_loss += loss.item()
                '''
                if counter % 10:
                    for name, p in discriminator.named_parameters():
                        if p.grad is not None:
                            disc_grad_sums[name] += p.grad.abs()
                    disc_batches += 1
                '''

            #discriminator.eval()
            if counter % 1 == 0:
                # 1) capture real-batch feature moments
                with torch.no_grad():
                    real_feats = discriminator.features(real_images)   # [B,512]
                    real_mean = real_feats.mean(dim=0)                     # [512]
                    real_std  = real_feats.std(dim=0)                      # [512]

                ##################### 
                # train the generator
                #generator.train()
                z = torch.randn(batch_size, noise_dim, 1, 1).to(device)
                fake_images = generator(z)

                div_loss = diversity_loss_pixel(fake_images)
                # trick the discriminator, smooth lables here also?
                g_labels = torch.ones(batch_size, 1, dtype=torch.float).to(device)
                #g_labels = torch.full((batch_size, 1), 0.9, device=device)
                
                generator_optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    g_output = discriminator(fake_images)
                    # TO DO: evaluate mean and std of self.gap values in discriminator on fakes
                    # and compare with mean and std of self.gap values in discriminator on reals
                    # penalize these distributions differ too much in batch sample 
                    g_loss = criterion(g_output, g_labels) + p / (div_loss + eps)
                
                with autocast(device_type='cuda'):
                    # adversarial
                    g_output = discriminator(fake_images)
                    adv_loss = criterion(g_output, g_labels)

                    fake_feats = discriminator.features(fake_images)   # [B,512]
                    fake_mean = fake_feats.mean(dim=0)
                    fake_std  = fake_feats.std(dim=0)

                    mean_diff = torch.mean((fake_mean - real_mean) ** 2)
                    std_diff  = torch.mean((fake_std  - real_std ) ** 2)
                    gap_pen   = lam_gap * (mean_diff + std_diff)

                    g_loss = adv_loss + gap_pen
                    #p / (div_loss + eps) + gap_pen

                generator_scaler.scale(g_loss).backward()
                generator_scaler.step(generator_optimizer)
                generator_scaler.update()
                generator_total_loss += g_loss.item()
                '''
                if counter % 10:
                    for name,p in generator.named_parameters():
                        if p.grad is not None:
                            gen_grad_sums[name] += p.grad.abs()
                    gen_batches += 1
                '''
                #generator.eval()

            counter += 1

            if counter % 20 == 0:
                with torch.no_grad():
                    z = torch.randn(16, noise_dim, 1, 1).to(device)
                    fakes = generator(z)
                    print("GENERATED SHAPE:")
                    print(fakes.shape)
                    fakes = fakes.to(torch.float32)
                    fakes = (fakes * std_tensor + mean_tensor).clamp(0, 1)
                    grid = torchvision.utils.make_grid(fakes.cpu(), nrow=4)
                    torchvision.utils.save_image(grid, f"./samples/epoch_{epoch}.png")
                    
                    fakes = generator(fixed_z)
                    fakes = fakes.to(torch.float32)
                    fakes = (fakes * std_tensor + mean_tensor).clamp(0, 1)
                    grid = torchvision.utils.make_grid(fakes.cpu(), nrow=4)
                    torchvision.utils.save_image(grid, f"./samples/epoch_fixed_{epoch}.png")

                    var = fakes.var(dim=0).mean().item()
                    print(f"Image variance (diversity): {var:.4f}")
                    
                    #################################
                    # evaluate accuracy of the discriminator
                    z = torch.randn(batch_size, noise_dim, 1, 1).to(device)
                    fakes = generator(z)
                    real_images = images.to(device)

                    real_logits = discriminator(real_images)
                    fake_logits = discriminator(fakes)

                    real_probs = torch.sigmoid(real_logits)
                    fake_probs = torch.sigmoid(fake_logits)

                    print("REAl:")
                    print(real_probs[0])
                    print("FAKE:")
                    print(fake_probs[0])

                    real_preds = (real_probs > 0.5).float()
                    fake_preds = (fake_probs > 0.5).float()

                    real_labels = torch.ones_like(real_preds)
                    fake_labels = torch.zeros_like(fake_preds)

                    real_acc = (real_preds == real_labels).float().mean().item()
                    fake_acc = (fake_preds == fake_labels).float().mean().item()

                    print("Discriminator Accuracy:")
                    print(f"  Reals: {real_acc:.2%}")
                    print(f"  Fakes: {fake_acc:.2%}")

                    print(f"Batch {counter}\n Generator Batch Loss: {g_loss}\n Discriminator Batch Loss: {d_loss}\n")
                    '''
                    gen_means  = {n: s/gen_batches  for n,s in gen_grad_sums.items()}
                    disc_means = {n: s/disc_batches for n,s in disc_grad_sums.items()}
                    plot_layer_heatmaps(gen_means,  f"Gen grads epoch {epoch}",  f"./samples/epoch_gen_{epoch}.png")
                    plot_layer_heatmaps(disc_means, f"Disc grads epoch {epoch}", f"./samples/epoch_disc_{epoch}.png")
                    '''
        end_time = time.time()

        print(f"Epoch {epoch + 1}\n Generator Training Loss: {generator_total_loss}\n Discriminator Training Loss: {discriminator_total_loss}\n Time: {end_time - start_time}s")
        print()
        torch.save(copy.deepcopy(discriminator.state_dict()), f"./models/checkpoint/discriminator.pth")
        torch.save(discriminator_optimizer.state_dict(), f"./models/checkpoint/d_optimizer.pth")
        torch.save(copy.deepcopy(generator.state_dict()), f"./models/checkpoint/generator.pth")
        torch.save(generator_optimizer.state_dict(), f"./models/checkpoint/g_optimizer.pth")
    
    
if __name__ == "__main__":
    main()