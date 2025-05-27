import torch
import math

class Head(torch.nn.Module):
    def __init__(self, d_embedding = 128, d_hidden = 128):
        super(Head, self).__init__()
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden

        self.W_Q = torch.nn.Linear(d_embedding, d_hidden)
        self.W_K = torch.nn.Linear(d_embedding, d_hidden)
        self.W_V = torch.nn.Linear(d_embedding, d_hidden)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        return torch.softmax( Q @ K.transpose(1, 2) / (self.d_hidden ** 0.5), dim = -1) @ V
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads = 8, d_embedding = 128, d_hidden = 128):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden

        self.heads = torch.nn.ModuleList([Head(d_embedding, d_hidden) for _ in range(n_heads)])
        self.W_O = torch.nn.Linear(n_heads * d_hidden, d_embedding)

        self.LayerNorm = torch.nn.LayerNorm(d_embedding)

    def forward(self, x):
        attention = self.W_O(torch.cat([head(x) for head in self.heads], dim = -1))
        return self.LayerNorm(attention + x)

class FeedForward(torch.nn.Module):
    def __init__(self, d_embedding = 128, d_hidden = 128):
        super(FeedForward, self).__init__()
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden

        self.W_1 = torch.nn.Linear(d_embedding, d_hidden)
        self.W_2 = torch.nn.Linear(d_hidden, d_embedding)

        self.LayerNorm = torch.nn.LayerNorm(d_embedding)

    def forward(self, x):
        s = self.W_1(x)
        s = torch.relu(s)
        s = self.W_2(s)
        return self.LayerNorm(s + x)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_embedding = 128, d_attention_hidden = 128, d_ffn_hidden = 128, n_heads = 8):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(n_heads, d_embedding, d_attention_hidden)
        self.ffn = FeedForward(d_embedding, d_ffn_hidden)

    def forward(self, x):
        x = self.mha(x)
        x = self.ffn(x)
        return x

class Transpose(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
      
class Unsqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, D = x.shape
        H, W = 8, 8
        assert D == H * W
        return x.view(B, C, H, W)
            

class ImageEncoder(torch.nn.Module):
    def __init__(self, d_embedding = 128, patch_size = 8, time_range = 1000, time_encoding_dim = 128):
        super(ImageEncoder, self).__init__()
        self.d_embedding = d_embedding
        self.patch_size = patch_size
        self.n_patches = (64 // patch_size) ** 2
        self.time_encoding_dim = time_encoding_dim
        self.time_range = time_range

        self.time_processing = torch.nn.Sequential(
            torch.nn.Linear(time_encoding_dim, time_encoding_dim * 2),
            torch.nn.LayerNorm(time_encoding_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(time_encoding_dim * 2, d_embedding),
            torch.nn.LayerNorm(d_embedding),
            torch.nn.ReLU()
        )

        self.patch_projection = torch.nn.Sequential(
            torch.nn.Conv2d(3, d_embedding, kernel_size = patch_size, stride = patch_size),
            torch.nn.Flatten(start_dim = 2, end_dim = 3),
            Transpose(1, 2),
            torch.nn.LayerNorm(d_embedding),
            torch.nn.ReLU()
        )

        self.positional_encoding = torch.nn.Parameter(torch.randn(1, self.n_patches, d_embedding), requires_grad = True)
    
    def time_encoding(self, t):
        sin = torch.cat([torch.sin(2 * math.pi * t * mul / self.time_range).unsqueeze(1) for mul in range(1, self.time_encoding_dim // 2 + 1)], dim = 1)
        cos = torch.cat([torch.cos(2 * math.pi * t * mul / self.time_range).unsqueeze(1) for mul in range(1, self.time_encoding_dim // 2 + 1)], dim = 1)
        out = torch.cat([sin, cos], dim = 1)
        return out
    

    def forward(self, x, t):
        time_encoding = self.time_encoding(t)
        
        time_encoding = self.time_processing(time_encoding).unsqueeze(1)
        

        x = self.patch_projection(x)

        x = x + self.positional_encoding


        x = x + time_encoding
        return x
    

class DiffusionTransformer(torch.nn.Module):
    def __init__(
            self,
            d_embedding = 128,
            patch_size = 8,
            time_range = 1000,
            time_encoding_dim = 128,
            n_transformer_blocks = 6,
            d_attention_hidden = 128,
            d_ffn_hidden = 128,
            n_heads = 8
    ):
        self.patch_size = patch_size
        self.n_patches = (64 // patch_size) ** 2
        super(DiffusionTransformer, self).__init__()
        self.image_encoder = ImageEncoder(d_embedding, patch_size, time_range, time_encoding_dim)
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(d_embedding, d_attention_hidden, d_ffn_hidden, n_heads) for _ in range(n_transformer_blocks)
        ])

        self.image_decoder = torch.nn.Sequential(
            torch.nn.Linear(d_embedding, 3 * self.patch_size * self.patch_size),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x, t):
        x = self.image_encoder(x, t)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.image_decoder(x)

        B, N, patch_dim = x.shape
        H = W = 64
        P = self.patch_size

        assert patch_dim == P * P * 3, f"Expected patch_dim={P*P*3}, but got {patch_dim}"
        assert N == (H // P) * (W // P), f"Expected {H//P}x{W//P}={H*W//(P*P)} patches, but got {N}"

        # reshape: [B, 64, 192] → [B, 8, 8, 8, 8, 3]
        x = x.view(B, H // P, W // P, P, P, 3)

        # permute to: [B, 3, 8, 8, 8, 8]
        x = x.permute(0, 5, 1, 3, 2, 4)

        # reshape to: [B, 3, 64, 64]
        x = x.reshape(B, 3, H, W)


        return x

