import torch
import matplotlib.pyplot as plt

def display(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()

        # Obsługa różnych wymiarów
        if x.dim() == 4:
            x = x[0]  # np. (1, 3, 64, 64) → (3, 64, 64)
        elif x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)  # np. (1, 28, 28) → (28, 28)

        # Przeskaluj z (-3, 3) do (0, 1)
        x = (x + 3) / 6.0
        x = torch.clamp(x, 0, 1)

        # Konwersja kanałów: (C, H, W) → (H, W, C)
        if x.dim() == 3:
            x = x.permute(1, 2, 0).numpy()

        # Wybór mapy kolorów
        if x.shape[-1] == 1:
            plt.imshow(x[:, :, 0], cmap="gray")
        else:
            plt.imshow(x)

        plt.axis('off')
        plt.show()

    else:
        raise TypeError("Wejście musi być tensorem PyTorch.")