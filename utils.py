import matplotlib.pyplot as plt
import torch




def display(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.dim() == 4 and x.shape[1] == 1:
            x = x[0, 0]  # np. (1, 1, 28, 28) → (28, 28)

        # Przeskaluj z (-3, 3) do (0, 1)
        x = (x + 3) / 6.0
        x = torch.clamp(x, 0, 1)

        plt.imshow(x.numpy(), cmap="gray")
        plt.axis('off')
        plt.show()
    else:
        raise TypeError("Wejście musi być tensorem PyTorch.")
