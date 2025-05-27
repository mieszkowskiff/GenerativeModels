import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from components import UNet_Tranformer

# === 1. Parametry ===
batch_size = 128
num_epochs = 3
learning_rate = 2e-4
timesteps = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 2. Rozkład szumu (marginal_prob_std) ===
def marginal_prob_std(t, sigma=25.):
    return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * torch.log(torch.tensor(sigma))))

# === 3. Ładowanie danych ===
transform = transforms.Compose([
    transforms.ToTensor(),  # skaluje do [0, 1]
    lambda x: x * 2. - 1.   # skaluje do [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# === 4. Inicjalizacja modelu ===
model = UNet_Tranformer(marginal_prob_std=marginal_prob_std).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# === 5. Funkcja treningowa ===
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.
    
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x, y = x.to(device), y.to(device)

        # 5.1 Losowy krok czasu dla każdego elementu w batchu
        t = torch.randint(low=1, high=timesteps, size=(x.size(0),), device=device)
        t_norm = t.float() / timesteps  # skaluje t do [0, 1]

        # 5.2 Szum i zaszumiony obraz
        z = torch.randn_like(x)
        std = marginal_prob_std(t_norm).view(-1, 1, 1, 1)
        x_noisy = x + std * z

        # 5.3 Predykcja szumu
        z_pred = model(x_noisy, t, y=y)

        # 5.4 MSE loss między prawdziwym a przewidzianym szumem
        loss = F.mse_loss(z_pred, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")