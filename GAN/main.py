import torch
import time
import tqdm
import copy
import shutil
import numpy as np

from torchsummary import summary
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import utils.components as components
import utils.utils as utils
from utils.architecture import Generator
from utils.architecture import Discriminator

class_list = ["fake", "real"]
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 30

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4837442934513092, 0.4360327422618866, 0.38709232211112976], 
            std=[0.19943906366825104, 0.19818559288978577, 0.19755707681179047]
        )
    ])

    train_dataset = ImageFolder(root='../dataset', transform=train_transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)

    generator = Generator()
    generator.to(device)
    discriminator = Discriminator()
    discriminator.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.0001)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr = 0.001)

    for epoch in range(epochs):
        print(f"Using device: {device}")
        start_time = time.time()

        discriminator.train()
        discriminator_scaler = GradScaler()
        discriminator_total_loss = 0
        
        for images, labels in tqdm.tqdm(train_loader):
            permutation = np.random.permutation(2*batch_size)
            # fake
            seeds = [torch.rand(100).reshape(-1, 1, 1) for _ in range(batch_size)].to(device)
            fake_images = [generator(it) for it in seeds].to(device)
            fake_labels = [1 for _ in range(batch_size)].to(device)
            
            # real
            real_images, real_labels = images.to(device), labels.long().to(device)
            print(real_images.shape)
            
            device_images = 0
            device_labels = 0

            discriminator_optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = discriminator(device_samples)
                loss = criterion(outputs, device_labels)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
                    
        end_time = time.time()

        epoch_f1 = components.evaluate_f1_score(model=model, test_loader=test_loader, device=device)
        F1_history[epoch] = epoch_f1
        loss_history[epoch] = total_loss
        time_history[epoch] = end_time - start_time

        print(f"Time for testing: {time.time() - end_time}")
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, F1_scrore: {epoch_f1}, Time: {end_time - start_time}s")
        print()
        if(epoch_f1>best_F1):
            torch.save(copy.deepcopy(model.state_dict()), f"./models/checkpoint/model.pth")
            torch.save(optimizer.state_dict(), f"./models/checkpoint/optimizer.pth")
            best_F1 = epoch_f1
    
    print("F1_score:")
    print(F1_history)
    print("Loss:")
    print(loss_history)
    print("Time:")
    print(time_history)
    print()

    print("Saving history data to history.txt")
    utils.save_txt(F1_history, "./training_data/F1_history.txt")
    utils.save_txt(loss_history, "./training_data/loss_history.txt")
    utils.save_txt(time_history, "./training_data/time_history.txt")
    print(f"Data gathered. Training performed succesfully.")

    # at the end of the training, type the name of the model, it will move the best model instance 
    # from chechpoint to models directory and name the model and optimizer files accordingly to the name 
    filename = input("Enter the model name to save the model and optimizer: ")
    model_name = filename
    opt_name = filename + "_optim"
    shutil.move("./models/checkpoint/model.pth", f"./models/{model_name}.pth")
    shutil.move("./models/checkpoint/optimizer.pth", f"./models/{opt_name}.pth")

    
if __name__ == "__main__":
    main()